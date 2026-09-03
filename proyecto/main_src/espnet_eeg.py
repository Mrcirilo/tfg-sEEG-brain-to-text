"""
espnet_eeg.py
=============
Adaptacion de ESPnet / OWSM v3.1 a senales neuronales intracorticales.

Este modulo contiene UNICAMENTE las piezas de ESPnet que hay que modificar para
que un modelo de reconocimiento de habla acepte actividad neuronal en lugar de
audio. Todo lo demas --bucle de entrenamiento, optimizador, perdida CTC,
decodificacion, calculo de metricas-- es ESPnet sin tocar, llamado desde el
notebook.

Cada bloque documenta tres cosas:

    [ESPnet]  lo que hace la libreria original
    [Cambio]  lo que se modifica
    [Motivo]  por que es necesario

Referencias al codigo fuente (espnet 202511):
    espnet2/s2t/espnet_model.py                          -> encode(), _extract_feats()
    espnet2/asr/encoder/e_branchformer_encoder.py:483    -> el isinstance() de la mascara
    espnet/nets/pytorch_backend/transformer/subsampling.py -> Conv2dSubsampling

Kiril · TFG · Brain-to-Text 2025
"""

import os
import re
import string
from itertools import groupby

import h5py
import numpy as np
import torch

try:                                    # espnet <= 202511
    from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
except ModuleNotFoundError:             # espnet >= 202609: el namespace v1 pasó a legacy
    from espnet2.legacy.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling


# =====================================================================
#  1. LECTURA DEL CORPUS
# =====================================================================
# [ESPnet]  espera un scp/wav.scp con rutas a ficheros de audio, que carga con
#           soundfile y entrega como forma de onda 1-D a 16 kHz.
# [Cambio]  se lee HDF5 con matrices (T, 512) ya extraidas a 50 Hz.
# [Motivo]  el dataset B2T'25 no distribuye audio: distribuye potencia de banda
#           y tasa de cruces por umbral de 256 electrodos (2 rasgos x 256).

def decode_transcription(arr):
    """Las transcripciones vienen como vector de codigos ASCII con relleno a 0."""
    arr = np.asarray(arr).ravel()
    return "".join(chr(int(x)) for x in arr if int(x) != 0)


def cargar_sesion(ruta_hdf5):
    """Devuelve la lista de trials de un fichero data_{train,val}.hdf5."""
    trials = []
    with h5py.File(ruta_hdf5, "r") as f:
        for clave in sorted(f.keys()):
            t = f[clave]
            trials.append({
                "input_features": t["input_features"][:],          # (T, 512) float32
                "text_raw": decode_transcription(t["transcription"][()]),
            })
    return trials


def cargar_split(raiz, sesiones, split="train", verbose=True):
    """Concatena varias sesiones. Las sesiones sin fichero de ese split se avisan."""
    salida = []
    for s in sesiones:
        ruta = os.path.join(raiz, s, f"data_{split}.hdf5")
        if not os.path.exists(ruta):
            if verbose:
                print(f"  [aviso] no existe {s}/data_{split}.hdf5, se salta")
            continue
        trials = cargar_sesion(ruta)
        if verbose:
            print(f"  {s}/{split}: {len(trials):4d} trials")
        salida.extend(trials)
    return salida


# =====================================================================
#  2. NORMALIZACION
# =====================================================================
# [ESPnet]  model.normalize es un GlobalMVN: resta media y divide por desviacion
#           tipica calculadas sobre TODO el corpus de audio de preentrenamiento.
# [Cambio]  z-score por trial y por canal, aplicado en el data_info.
# [Motivo]  la ganancia de cada electrodo deriva entre sesiones a lo largo de los
#           20 meses de grabacion. Una estadistica global mezclaria sesiones que
#           no son comparables entre si; una estadistica por trial no.

def znorm(x):
    x = x.astype(np.float32)
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)


def remove_punctuation(texto):
    return texto.translate(str.maketrans("", "", string.punctuation))


# =====================================================================
#  3. DATASET
# =====================================================================
# [ESPnet]  ESPnetEZDataset espera que __getitem__ devuelva un diccionario cuyas
#           claves consuma luego el data_info.
# [Cambio]  ninguno estructural. Se devuelven los cuatro campos que pide la tarea
#           s2t (speech, text, text_prev, text_ctc) con la senal en lugar del audio.
# [Motivo]  mantener el contrato de ESPnet-EZ intacto es justamente lo que permite
#           reutilizar su Trainer sin escribir un bucle de entrenamiento propio.

class SeeGDataset(torch.utils.data.Dataset):

    def __init__(self, trials, idioma="eng"):
        self.trials = trials
        self.idioma = idioma

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        d = self.trials[idx]
        texto = d["text_raw"].lower()
        return {
            "input_features": d["input_features"],
            # el prompt de OWSM: idioma + tarea + sin marcas de tiempo
            "text":      f"<{self.idioma}><asr><notimestamps> {texto}",
            "text_prev": "<na>",
            "text_ctc":  remove_punctuation(texto),
            "text_raw":  d["text_raw"],
        }


# =====================================================================
#  4. EL FRONTEND
# =====================================================================
# [ESPnet]  encoder.embed = Conv2dSubsampling(80, 384, ...). Recibe el banco de
#           filtros mel de 80 dimensiones y lo proyecta a las 384 del encoder,
#           dividiendo la resolucion temporal entre 4.
# [Cambio]  se reconstruye la misma clase con idim=512.
# [Motivo]  nuestra entrada tiene 512 rasgos, no 80. Es literalmente el punto
#           donde el modelo deja de ser un modelo de audio.
#
# NOTA IMPORTANTE sobre la herencia (esto costo un dia entero de depuracion):
# el encoder decide si pasar la mascara de relleno al embed con un isinstance
# contra las clases de submuestreo de ESPnet (e_branchformer_encoder.py:483).
# Si un frontend propio NO hereda de Conv2dSubsampling, ESPnet cae en la rama
# `elif self.embed is not None: xs_pad = self.embed(xs_pad)` y lo llama SIN
# mascara, con lo que revienta con:
#     TypeError: forward() missing 1 required positional argument: 'x_mask'
# De ahi que FrontendLineal herede de Conv2dSubsampling sin llamar a su __init__.

def crear_frontend(tipo, idim, odim, pos_enc):
    """Devuelve el modulo que sustituye a encoder.embed."""

    if tipo == "conv2d":
        # La adaptacion minima: la clase de ESPnet, sin modificar, con idim=512.
        # Ojo al coste: la capa de salida es
        #     Linear(odim * ((idim-1)//2 - 1)//2, odim)
        # que con idim=80 son 384*19 = 7.296 entradas y con idim=512 son
        # 384*127 = 48.768. El frontend pasa de 2,8 M a 18,7 M de parametros.
        return Conv2dSubsampling(idim, odim, dropout_rate=0.0, pos_enc=pos_enc)

    if tipo == "lineal":
        # Alternativa: proyeccion densa 512 -> 384 y decimacion temporal.
        # La conv2d convoluciona a lo largo del eje de los 512 rasgos como si
        # fuera un eje de frecuencias, es decir, asume que los electrodos
        # contiguos estan relacionados. En un Utah array el orden de los
        # electrodos es arbitrario, asi que esa suposicion no se sostiene.
        class FrontendLineal(Conv2dSubsampling):
            def __init__(self, factor=4):
                torch.nn.Module.__init__(self)   # NO llamar a Conv2dSubsampling.__init__
                self.proj = torch.nn.Linear(idim, odim)
                self.pos_enc = pos_enc
                self.factor = factor

            def forward(self, x, x_mask):
                x = self.proj(x)[:, ::self.factor, :]
                t = x.size(1)
                x = self.pos_enc(x)              # puede devolver (x, pos_emb)
                if x_mask is not None:
                    x_mask = x_mask[:, :, ::self.factor][:, :, :t]
                return x, x_mask

        return FrontendLineal()

    raise ValueError(f"Frontend desconocido: {tipo}")


# =====================================================================
#  5. LA ADAPTACION
# =====================================================================
# Cuatro cambios sobre el modelo ya construido por S2TTask.build_model().
# Es todo lo que separa un modelo de habla de uno de senal neuronal.

def adaptar_a_seeg(model, idim=512, frontend="conv2d", verbose=True):
    """Convierte un ESPnetS2TModel de audio en uno que acepta (T, idim) directo."""

    antes = {
        "frontend":  type(model.frontend).__name__,
        "specaug":   type(model.specaug).__name__,
        "normalize": type(model.normalize).__name__,
        "embed_in":  model.encoder.embed.out[0].in_features,
    }

    # --- 1. frontend acustico -----------------------------------------
    # [ESPnet]  DefaultFrontend: STFT -> banco de filtros mel -> 80 dims.
    # [Cambio]  None.
    # [Motivo]  con frontend=None, _extract_feats() devuelve `speech` tal cual
    #           (espnet_model.py:400). Nuestras caracteristicas ya vienen
    #           extraidas del cortex motor; no hay forma de onda que analizar.
    model.frontend = None

    # --- 2. SpecAugment -----------------------------------------------
    # [ESPnet]  enmascara bandas de frecuencia y de tiempo del espectrograma.
    # [Cambio]  None.
    # [Motivo]  el enmascarado de frecuencia asume que el eje de rasgos es un
    #           eje de frecuencias ordenado. Aqui es un indice de electrodo.
    model.specaug = None

    # --- 3. normalizacion global --------------------------------------
    # [ESPnet]  GlobalMVN con estadisticas del corpus de audio.
    # [Cambio]  None; se sustituye por z-score por trial en el data_info.
    # [Motivo]  ver el bloque 2.
    model.normalize = None

    # --- 4. embed del encoder -----------------------------------------
    # [ESPnet]  Conv2dSubsampling(80, 384).
    # [Cambio]  el mismo modulo con idim=512, reutilizando la codificacion
    #           posicional preentrenada.
    # [Motivo]  es la unica capa del modelo cuya dimension de entrada depende
    #           de la modalidad. El resto del encoder trabaja ya en 384.
    pos_enc = model.encoder.embed.out[1]
    odim = model.encoder.output_size()
    model.encoder.embed = crear_frontend(frontend, idim, odim, pos_enc)

    despues = {
        "frontend":  type(model.frontend).__name__,
        "specaug":   type(model.specaug).__name__,
        "normalize": type(model.normalize).__name__,
        "embed_in":  (model.encoder.embed.out[0].in_features
                      if hasattr(model.encoder.embed, "out")
                      else model.encoder.embed.proj.in_features),
    }

    if verbose:
        print(f"{'componente':<14}{'ESPnet original':<26}{'adaptado':<26}")
        print("-" * 66)
        for k in antes:
            print(f"{k:<14}{str(antes[k]):<26}{str(despues[k]):<26}")

    return model


def contar_parametros(model):
    total = sum(p.numel() for p in model.parameters())
    entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, entrenables


# =====================================================================
#  6. INFERENCIA Y METRICAS
# =====================================================================
# [ESPnet]  Speech2Text hace busqueda en haz conjunta CTC-atencion sobre audio.
# [Cambio]  decodificacion CTC voraz llamando a model.encode() y
#           model.ctc.log_softmax(), que son metodos de ESPnet sin tocar.
# [Motivo]  con ctc_weight=1.0 el decodificador autorregresivo no participa en
#           la perdida, asi que la salida del sistema ES la de la rama CTC.
#           Ademas la voraz aisla la calidad del encoder, sin modelo de lenguaje.

@torch.no_grad()
def logits_ctc(model, caracteristicas, device):
    """Devuelve las log-probabilidades CTC (T', V) de un trial."""
    x = torch.as_tensor(znorm(caracteristicas), dtype=torch.float32, device=device)
    speech = x.unsqueeze(0)                                   # (1, T, 512)
    lengths = torch.tensor([speech.size(1)], device=device)
    enc, _ = model.encode(speech, lengths)
    if isinstance(enc, tuple):                                # InterCTC devuelve tupla
        enc = enc[0]
    return model.ctc.log_softmax(enc)[0]


def colapsar_ctc(ids, blank=0):
    """Funcion de colapso de CTC: quita repeticiones y luego los blancos."""
    return [int(k) for k, _ in groupby(ids) if int(k) != blank]


def decodificar_voraz(model, caracteristicas, device, tokenizer, converter, blank=0):
    """Devuelve (texto, tasa_de_blancos, ids_por_frame)."""
    logp = logits_ctc(model, caracteristicas, device)
    ids = logp.argmax(dim=-1).cpu().numpy()
    tasa_blanco = float((ids == blank).mean())
    tokens = converter.ids2tokens(colapsar_ctc(ids, blank))
    return tokenizer.tokens2text(tokens), tasa_blanco, ids


def normalizar_para_evaluar(texto):
    """Minusculas, sin simbolos especiales de OWSM, sin puntuacion, sin dobles espacios."""
    texto = texto.lower().replace("\x00", "")
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = remove_punctuation(texto)
    return re.sub(r"\s+", " ", texto).strip()


def tasa_error(referencia, hipotesis, unidad="caracter"):
    """CER o WER via editdistance, que es la misma dependencia que usa ESPnet."""
    import editdistance

    ref = normalizar_para_evaluar(referencia)
    hyp = normalizar_para_evaluar(hipotesis)
    if unidad == "palabra":
        ref, hyp = ref.split(), hyp.split()
    if len(ref) == 0:
        return float(len(hyp) > 0)
    return editdistance.eval(ref, hyp) / len(ref)
