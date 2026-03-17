import os
import matplotlib.pyplot as plt
from pgvector.psycopg import register_vector
import psycopg
import tempfile
import torch
import torchvision
from tqdm import tqdm

seed = False

# In terminal, run: export CONNECTION_STRING=postgres://[user]:password@host:port/database
connection_string  = os.environ['CONNECTION_STRING']

# establish connection
conn = psycopg.connect(connection_string)
conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

# load images
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = torchvision.datasets.CIFAR10(root=tempfile.gettempdir(), train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000)

# load pretrained model
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = torchvision.models.resnet18(weights='DEFAULT')
model.fc = torch.nn.Identity()
model.to(device)
model.eval()


def generate_embeddings(inputs):
    return model(inputs.to(device)).detach().cpu().numpy()

# generate and store embeddings
if seed:
    conn.execute('DROP TABLE IF EXISTS test_images')
    conn.execute('CREATE TABLE test_images (id bigserial PRIMARY KEY, embedding vector(512))')
    conn.commit()

    print('Generating embeddings')
    for data in tqdm(dataloader):
        embeddings = generate_embeddings(data[0])

        sql = 'INSERT INTO test_images (embedding) VALUES ' + ','.join(['(%s)' for _ in embeddings])
        params = [embedding for embedding in embeddings]
        conn.execute(sql, params)
    conn.commit()

# load 5 random unseen images
queryset = torchvision.datasets.CIFAR10(root=tempfile.gettempdir(), train=False, download=True, transform=transform)
queryloader = torch.utils.data.DataLoader(queryset, batch_size=5, shuffle=True)
images = next(iter(queryloader))[0]

# generate and query embeddings
results = []
embeddings = generate_embeddings(images)
for image, embedding in zip(images, embeddings):
    result = conn.execute('SELECT id FROM test_images ORDER BY embedding <=> %s LIMIT 5', (embedding,)).fetchall()
    nearest_images = [dataset[row[0] - 1][0] for row in result]
    results.append([image] + nearest_images)

fig, axs = plt.subplots(len(results), len(results[0]))
for i, result in enumerate(results):
    for j, image in enumerate(result):
        ax = axs[i, j]
        ax.imshow((image / 2 + 0.5).permute(1, 2, 0).numpy())
        ax.set_title(f"Query Image {i+1}" if j == 0 else f"Result {j}")
        ax.set_axis_off()
plt.show(block=True)
