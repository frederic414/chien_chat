from fastapi import APIRouter, UploadFile
import os
from services.image_processing import preprocess_image, predict_image, generate_unique_filename
router = APIRouter()

# Chemin relatif du dossier dans votre projet
upload_dir = "storage"

# Vérifier si le dossier existe, sinon le créer
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)
@router.get("/")
async def root():
    return {'Message': 'Hello word!'}

@router.post("/predict")
async def predict(image_file: UploadFile):
    filename = generate_unique_filename(image_file.filename)
    file_path = os.path.join(upload_dir, filename)
    with open(file_path, "wb") as file:
        contents = await image_file.read()
        file.write(contents)
    image = preprocess_image(file_path)
    prediction = predict_image(image)
    # Supprimer le fichier après le traitement
    os.remove(file_path)

    return {"prediction": prediction}
