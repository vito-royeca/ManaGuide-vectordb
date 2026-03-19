import os
import sys

import psycopg
from pgvector.psycopg import register_vector
from PIL import Image
import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

# In terminal, run: export CONNECTION_STRING=postgres://[user]:[password]@[host]:[port]/[database]
connection_string  = os.environ['CONNECTION_STRING']

model = None
preprocess = None
connection = None

# Load pre-trained ResNet model
def init_model():
    global model
    global preprocess

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()
    # Remove the final classification layer to get embeddings
    model = torch.nn.Sequential(*list(model.children())[:-1])

    preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Connect to PostgreSQL database using connection string
def init_database():
    global connection
    connection = psycopg.connect(connection_string)
    register_vector(connection)

def create_embeddings(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(image_tensor)
    return embedding.squeeze().numpy()

def create_cmimage(path, card_id):
    cursor = connection.cursor()

    cursor.execute("SELECT cmcard FROM cmimage WHERE cmcard = (%s)", (card_id,))
    result = cursor.fetchone()

    if result is None or len(result) == 0:
        # check if card exists in cmcard table before inserting
        cursor.execute("SELECT new_id FROM cmcard WHERE new_id = (%s)", (card_id,))
        result = cursor.fetchone()

        if result is not None:
            print(f"Inserting... {card_id}")
            embeddings = create_embeddings(path)
            cursor.execute("INSERT INTO cmimage (cmcard, embeddings) VALUES (%s, %s)", (card_id, embeddings))
            connection.commit()
    else:
        print(f"Already exists... {card_id}")
    cursor.close()

def optimize_database():
    cursor = connection.cursor()
    cursor.execute("DROP INDEX IF EXISTS cmcard_index_embeddings;")
    cursor.execute("CREATE INDEX cmcard_index_embeddings ON cmimage USING ivfflat (embeddings vector_l2_ops) WITH (lists = 100);")
    connection.commit()

def process_images(image_path, card_id=None):
    if os.path.isdir(image_path):
        for root, dirs, files in os.walk(image_path):
            for file in files:
                if file == "normal.jpg":
                    path = os.path.join(root, file)
                    card_id = path.split(os.path.sep)[-4] + "_" + path.split(os.path.sep)[-3] + "_" + path.split(os.path.sep)[-2]
                    create_cmimage(path, card_id)

    else:
        create_cmimage(image_path, card_id)

def main(image_path, card_id=None):
    init_model()
    init_database()
    process_images(image_path, card_id)
    optimize_database()
    connection.close()

# __name__
if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate.py <image_path> <card_id>")
        sys.exit(1)

    image_path = sys.argv[1]
    card_id = None

    if len(sys.argv) >= 3:
        card_id = sys.argv[2]
    main(image_path, card_id)