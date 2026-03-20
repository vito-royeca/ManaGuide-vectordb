import os
import sys

import psycopg
from pgvector.psycopg import register_vector
from PIL import Image
import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

from image_processor import ImageProcessor

# In terminal, run: export CONNECTION_STRING=postgres://[user]:[password]@[host]:[port]/[database]
connection_string  = os.environ['CONNECTION_STRING']

# image_base = "/mnt/managuide/images/cards/"
image_base = "/Users/vitoroyeca/workspace/ManaGuide/mount_images/cards"
model = None
preprocess = None
connection = None

# Load pre-trained ResNet model
def init_model():
    global model
    global preprocess

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()  # Set to evaluation mode
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
    # Register the vector type with psycopg2
    register_vector(connection)

def create_embeddings(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embeddings = model(image_tensor)
    return embeddings.squeeze().numpy()

def search_images(embeddings, limit=3):
    cursor = connection.cursor()
    
    # Convert query_embedding to a list to store in PostgreSQL
    embedding_list = embeddings.tolist()
    query =  f"SELECT cmcard, embeddings <-> '{embedding_list}' AS distance FROM cmimage ORDER BY embeddings <-> '{embedding_list}' LIMIT {limit};"
    cursor.execute(query)
    
    similar_images = cursor.fetchall()
    cursor.close()
    
    return similar_images

def process_image(image_path):
    processor = ImageProcessor(image_path)
    return processor.process_image()

def show_results(image_path):
    rows = []

    if os.path.isdir(image_path):
        for root, dirs, files in os.walk(image_path):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                # if file == "normal.jpg":
                # if file != "art_crop.jpg" and file != "png.jpg":
                    path = os.path.join(root, file)
                    path = process_image(path)
                    query = create_embeddings(path)
                    results = search_images(query)
                    rows.append((path, results))
    else:
        path = process_image(image_path)
        query = create_embeddings(path)
        results = search_images(query)
        rows.append((path, results))

    if len(rows) == 0:
        print("No results found.")
        return

    # render_plot(rows)
    render_html(rows)

def render_html(rows):
    html = "<html><body><table border=\"1\"><tr><th>Query Image</th><th>Result 1</th><th>Result 2</th><th>Result 3</th></tr>"

    for row in rows:
        path = row[0]
        results = row[1]
        
        html += "<tr>"

        html += f"<td><p>{os.path.basename(path)}</p><p><img src='{path}' width='200'></p></td>"

        for result in results:
            card_id = result[0]
            distance = result[1]
            
            card_path = card_id
            index = card_path.find("_")
            card_path = card_path[:index] + "/" + card_path[index+1:]

            index = card_path.find("_")
            card_path = card_path[:index] + "/" + card_path[index+1:]

            card_path = os.path.join(image_base, card_path, "normal.jpg")

            html += f"<td><p>{card_id}&nbsp;&nbsp;Distance: {distance:.2f}</p><p><img src='{card_path}' width='200'></p></td>"
        html += "</tr>"
    
    html += "</table></body></html>"
    
    with open("results.html", "w") as f:
        f.write(html)

def main(image_path):
    init_model()
    init_database()
    show_results(image_path)
    connection.close()

# __name__
if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Usage: python search.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    main(image_path)
