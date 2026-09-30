import numpy as np
import onnxruntime as ort

# 1. Charger la session ONNX
session = ort.InferenceSession("./onnx/model.onnx")

# 2. Récupérer automatiquement le nom de l'entrée
input_name = session.get_inputs()[0].name

# 3. Préparer une donnée de test (au format float32)
data = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)

# 4. Lancer l'inférence
outputs = session.run(None, {input_name: data})

# 5. Récupérer les résultats
# outputs[0] contient la classe prédite, outputs[1] les probabilités par classe
label = outputs[0]
probabilities = outputs[1]

print("Classe prédite :", label)
print("Probabilités :", probabilities)
