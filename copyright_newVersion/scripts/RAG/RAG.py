import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
from tqdm import tqdm  

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
model = model.to(device)

references = [
    "of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole,",
    "hair like the stuff on their heads (which is curly); have long clever brown fingers, good-natured faces,",
    "of this hobbit—of Bilbo Baggins, that is—was the famous Belladonna Took, one of the three remarkable daughters of the Old Took, head of the hobbits who lived across The Water,",
    "of pressure that you cannot withstand, even if you wished to. You will do what is required of you.' 'But what is it, what is it? How can I do it if I don't know what it is?' O'Brien picked up the cage and brought it across to",
    "? What are you doing here? What time did you leave work? Is this your usual way home?'--and so on and so forth. Not that there was any rule against walking home by an unusual route: but it was enough to draw attention to you if the Thought Police heard",
    "turned against Goldstein at all, but, on the contrary, against Big Brother, the Party, and the Thought Police; and at such moments his heart went out to the lonely, derided heretic on the screen, sole guardian of truth and sanity in a world of lies. And yet the very next",
    ". Tyrion danced back in while the brigand's leg was still pinned beneath his fallen mount, and buried the axe in the man's neck, just above the shoulder blades. As he struggled to yank the blade loose, he heard Marillion moaning under the bodies. \"Someone help me,\" the singer"
    ]

def batch_encode_references(model, references, batch_size=8):
    all_vectors = []
    for i in tqdm(range(0, len(references), batch_size), desc="Encoding references"):
        batch = references[i:i + batch_size]
        batch_vectors = model.encode(batch, convert_to_tensor=True, device=device)
        all_vectors.append(batch_vectors.cpu().numpy())
    return np.vstack(all_vectors)

references = [
    "of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole,",
    "hair like the stuff on their heads (which is curly); have long clever brown fingers, good-natured faces,",
    "of this hobbit—of Bilbo Baggins, that is—was the famous Belladonna Took, one of the three remarkable daughters of the Old Took, head of the hobbits who lived across The Water,",
    "of pressure that you cannot withstand, even if you wished to. You will do what is required of you.' 'But what is it, what is it? How can I do it if I don't know what it is?' O'Brien picked up the cage and brought it across to",
    "? What are you doing here? What time did you leave work? Is this your usual way home?'--and so on and so forth. Not that there was any rule against walking home by an unusual route: but it was enough to draw attention to you if the Thought Police heard",
    "turned against Goldstein at all, but, on the contrary, against Big Brother, the Party, and the Thought Police; and at such moments his heart went out to the lonely, derided heretic on the screen, sole guardian of truth and sanity in a world of lies. And yet the very next",
    ". Tyrion danced back in while the brigand's leg was still pinned beneath his fallen mount, and buried the axe in the man's neck, just above the shoulder blades. As he struggled to yank the blade loose, he heard Marillion moaning under the bodies. \"Someone help me,\" the singer"
    ]

def batch_encode_references(model, references, batch_size=8):
    all_vectors = []
    for i in tqdm(range(0, len(references), batch_size), desc="Encoding references"):
        batch = references[i:i + batch_size]
        batch_vectors = model.encode(batch, convert_to_tensor=True, device=device)
        all_vectors.append(batch_vectors.cpu().numpy())
    return np.vstack(all_vectors)


reference_vectors = batch_encode_references(model, references)

dimension = reference_vectors.shape[1]

nlist = 3
quantizer = faiss.IndexFlatL2(dimension)
gpu_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

print("Training the index...")
gpu_index.train(reference_vectors) 
print("Index training completed.")

print("Adding vectors to the index...")
gpu_index.add(reference_vectors)
print("Vectors added to the index.")



def search_next_sentence(input_text, top_k=1):
    print(f"Searching for next sentence for input: '{input_text}'...")
    input_vector = model.encode([input_text], convert_to_tensor=True, device=device).cpu().numpy()
    _, indices = gpu_index.search(input_vector, top_k)
    return [references[i] for i in indices[0]]

input_sentence = "foes right and left. Ser Rodrik hammered at the big man in the shadowskin cloak, their horses dancing round each other as they traded blow for blow. Jyck vaulted onto a horse and galloped bareback into the fray. Tyrion saw an arrow sprout from the throat of the man in the shadowskin cloak. When he opened his mouth to scream, only blood came out. By the time he fell, Ser Rodrik was fighting someone else. Suddenly Marillion shrieked, covering his head with his woodharp as a horse leapt over their rock. Tyrion scrambled to his feet as the A GAME OF THRONES 295 rider turned to come back at them, hefting a spiked maul. Tyrion swung his axe with both hands. The blade caught the charging horse in the throat with a meaty thunk, angling upward, and Tyrion almost lost his grip as the animal screamed and collapsed. He managed to wrench the axe free and lurch clumsily out of the way. Marillion was less fortunate. Horse and rider crashed to the ground in a tangle on top of the singer"
next_sentence = search_next_sentence(input_sentence)
print("Recommended next sentence:", next_sentence)