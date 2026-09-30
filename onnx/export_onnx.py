import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Dépendances pour la conversion ONNX
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

# 1. Données d'exemple et entraînement
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modèle LightGBM
model = lgb.LGBMClassifier(n_estimators=20, random_state=42)
model.fit(X_train, y_train)

# 2. Définition du format d'entrée ONNX
# [None, 4] = batch dynamique (nombre de lignes variable), 4 features
initial_type = [("float_input", FloatTensorType([None, 4]))]

# 3. Conversion du modèle LightGBM vers ONNX
onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_type)

# 4. Sauvegarde dans un fichier .onnx
with open("./onnx/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Modèle LightGBM exporté avec succès dans 'model.onnx' !")
