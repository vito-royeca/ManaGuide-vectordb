import os
import sys
import time
from datetime import datetime
from timeit import default_timer as timer
from datetime import timedelta

import psycopg
from pgvector.psycopg import register_vector
from PIL import Image
from psycopg import sql
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
    file_timestamp = datetime.fromtimestamp(os.path.getmtime(path)).astimezone(tz=None)
    cursor = connection.cursor()

    query = t"SELECT i.cmcard, i.date_updated FROM cmimage i LEFT JOIN cmcard c ON i.cmcard = c.new_id WHERE cmcard = {card_id}"
    cursor.execute(query)
    result = cursor.fetchone()

    if result is None or len(result) == 0:
        # check if card exists in cmcard table before inserting
        query = t"SELECT new_id FROM cmcard WHERE new_id = {card_id}"
        cursor.execute(query)
        result = cursor.fetchone()

        if result is not None:
            print(f"Inserting... {card_id}")
            embeddings = create_embeddings(path)
            query = t"INSERT INTO cmimage (cmcard, embeddings, date_updated) VALUES ({card_id}, {embeddings}, {file_timestamp})"
            cursor.execute(query)
            connection.commit()
    else:
        db_timestamp = result[1].astimezone(tz=None)
        if file_timestamp != db_timestamp:
            print(f"mismatch -  file: {file_timestamp} vs db: {db_timestamp} for card {card_id}")
            embeddings = create_embeddings(path)
            query = t"UPDATE cmimage SET embeddings = {embeddings}, date_updated = {file_timestamp} WHERE cmcard = {card_id}"
            cursor.execute(query)
            connection.commit()
        else:
            print(f"Already exists... {card_id}")
    cursor.close()

def batch_process_cmimages(path_dictionary):
    cursor = connection.cursor()
    card_ids = path_dictionary.keys()
    found_card_values = {}
    missing_card_ids = []
    batch_insert_values = []
    batch_update_values = []
    
    query = sql.SQL("SELECT cmcard, date_updated FROM cmimage WHERE cmcard IN ({})").format(
        sql.SQL(',').join(map(sql.Literal, card_ids)))
    cursor.execute(query)
    for record in cursor:
        found_card_values[record[0]] = record[1].astimezone(tz=None)
    missing_card_ids = [card_id for card_id in card_ids if card_id not in found_card_values.keys()]

    # filter out unupdated cards, i.e compare file_timestamp vs. db_timestamp
    for k, v in found_card_values.items():
        file_name = path_dictionary[k]
        file_timestamp = datetime.fromtimestamp(os.path.getmtime(file_name)).astimezone(tz=None)

        if file_timestamp != v:
            print(f"\tmismatch -  file: {file_timestamp} vs db: {v} for card {k}")
            embeddings = create_embeddings(file_name)
            batch_update_values.append((embeddings, file_timestamp, k))

    # bulk update
    if len(batch_update_values) > 0:
        print(f"\tUpdating {len(batch_update_values)} rows...")
        query = """
            UPDATE cmimage i 
            SET embeddings = j.embeddings, date_updated = j.date_updated 
            FROM VALUES(%s) as j(cmcard, embeddings, date_updated)
            WHERE i.cmcard = j.cmcard;
        """
        ([psycopg.ClientCursor(connection).mogrify(query, params) for params in batch_update_values])
        cursor.executemany(query, params_seq=batch_update_values)
        connection.commit()

    # check if missing_card_ids are in cmcard table before inserting
    if len(missing_card_ids) > 0:
        query = sql.SQL("SELECT new_id FROM cmcard WHERE new_id IN ({})").format(
            sql.SQL(', ').join(map(sql.Literal, missing_card_ids)))
        cursor.execute(query)
        for record in cursor:
            file_name = path_dictionary[record[0]]
            file_timestamp = datetime.fromtimestamp(os.path.getmtime(file_name)).astimezone(tz=None)
            embeddings = create_embeddings(file_name)
            batch_insert_values.append((record[0], embeddings, file_timestamp))

    # bulk insert
    if len(batch_insert_values) > 0:
        print(f"\tInserting {len(batch_insert_values)} rows...")
        query = """
            INSERT INTO cmimage(cmcard, embeddings, date_updated) VALUES (%s, %s, %s)
        """
        ([psycopg.ClientCursor(connection).mogrify(query, params) for params in batch_insert_values])
        cursor.executemany(query, params_seq=batch_insert_values)
        connection.commit()

def optimize_database():
    cursor = connection.cursor()
    cursor.execute("DROP INDEX IF EXISTS cmcard_index_embeddings;")
    cursor.execute("CREATE INDEX cmcard_index_embeddings ON cmimage USING ivfflat (embeddings vector_l2_ops) WITH (lists = 100);")
    connection.commit()

def format_time(sec):
   return timedelta(seconds=sec)
   
def process_images(image_path, card_id=None):
    alpha = timer()

    if os.path.isdir(image_path):
        batch_count = 1000
        total_count = 0
        path_dictionary = {}

        for root, dirs, files in os.walk(image_path):
            for file in files:
                if file == "normal.jpg":
                    path = os.path.join(root, file)
                    card_id = path.split(os.path.sep)[-4] + "_" + path.split(os.path.sep)[-3] + "_" + path.split(os.path.sep)[-2]
                    path_dictionary[card_id] = path

                if len(path_dictionary.keys()) >= batch_count:
                    start = timer()
                    total_count += batch_count
                    batch_process_cmimages(path_dictionary)
                    path_dictionary.clear()
                    end = timer()
                    print(f"Processed {total_count} rows in {format_time(end-start)}")

        if len(path_dictionary.keys()) >= 1:
            batch_process_cmimages(path_dictionary)
        path_dictionary.clear()
        print(f"Total rows: {total_count}")
    else:
        create_cmimage(image_path, card_id)

    omega = timer()
    print(f"Total time: {format_time(omega-alpha)}")

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