"""Transfer learning-тің "басы" (head): мұздатылған модель белгілерінің үстіндегі классификатор.

article_head — GeeksforGeeks мақаласындағы бас (дәл сол қабаттар):
    GAP -> BatchNorm -> Dense(512, relu) -> Dense(256, relu) -> Dropout(0.5)
        -> Dense(128, relu) -> Dense(120, softmax)
    (GAP қабаты белгілерді шығарғанда қолданылған, сондықтан мұнда кіріс — GAP векторы.)

linear_head — біздің ұсынысымыз: Dropout -> Dense(120, softmax).
    Белгілер онсыз да күшті болғандықтан, параметрі аз қарапайым бас аз шамадан тыс үйренеді
    (overfitting) және ықтималдықтары шынайырақ болады.
"""
import keras
from keras import layers


def article_head(input_dim: int, num_classes: int = 120) -> keras.Model:
    inp = keras.Input(shape=(input_dim,), name="features")
    x = layers.BatchNormalization()(inp)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax", name="breed")(x)
    return keras.Model(inp, out, name="article_head")


def linear_head(input_dim: int, num_classes: int = 120, dropout: float = 0.5, l2: float = 1e-4) -> keras.Model:
    inp = keras.Input(shape=(input_dim,), name="features")
    x = layers.Dropout(dropout)(inp)
    out = layers.Dense(num_classes, activation="softmax", name="breed",
                       kernel_regularizer=keras.regularizers.l2(l2))(x)
    return keras.Model(inp, out, name="linear_head")


HEADS = {"article": article_head, "linear": linear_head}
