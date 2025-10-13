from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os, json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model
MODEL_PATH = r"C:\veni project\egg_classifier_model.h5"
model = load_model(MODEL_PATH)

# Load class labels
LABEL_PATH = r"C:\veni project\class_labels.json"
if os.path.exists(LABEL_PATH):
    with open(LABEL_PATH, "r") as f:
        label_map = json.load(f)
        labels = {v: k for k, v in label_map.items()}
else:
    labels = {i: f"Class_{i}" for i in range(model.output_shape[-1])}
    print("⚠️ class_labels.json not found — using placeholder class names.")

# Expanded Mini Encyclopedia with local image paths and scientific names
egg_info = {
    "chicken": {
        "scientific_name": "Gallus gallus domesticus",
        "color_size": "Color: White/Brown | Size: Small to Medium",
        "habitat": "Domestic farms worldwide",
        "fun_fact": "Chickens are the most common domestic bird in the world, laying over 1 trillion eggs annually.",
        "image": "images/chicken_egg.jpg",
        "nutrition": "High in protein, vitamins B12 and D.",
        "uses": "Common in baking, cooking, and as a breakfast staple."
    },
    "duck": {
        "scientific_name": "Anas platyrhynchos domesticus",
        "color_size": "Color: White/Greenish | Size: Medium",
        "habitat": "Ponds, wetlands, and farms",
        "fun_fact": "Duck eggs have a richer flavor and larger yolks than chicken eggs.",
        "image": "images/duck_egg.jpg",
        "nutrition": "Higher in fat and cholesterol, rich in omega-3s.",
        "uses": "Popular in Asian cuisine, like balut or salted eggs."
    },
    "ostrich": {
        "scientific_name": "Struthio camelus",
        "color_size": "Color: Cream/Ivory | Size: Very Large (up to 6 inches)",
        "habitat": "African savannas and farms",
        "fun_fact": "One ostrich egg can weigh as much as 24 chicken eggs and is the largest egg of any living bird.",
        "image": "images/ostrich_egg.jpg",
        "nutrition": "Lower in cholesterol than chicken eggs, high in iron.",
        "uses": "Eaten boiled or scrambled; shells used for crafts."
    },
    "\u00a0turkey": {
        "scientific_name": "Meleagris gallopavo",
        "color_size": "Color: White to creamy white with brown speckles | Size: Large (25-30% larger than duck eggs)",
        "habitat": "Forests, farms, and open woodlands in North America",
        "fun_fact": "Turkey eggs are rarer due to hens laying fewer eggs per year, making them a delicacy.",
        "image": "images/turkey_egg.jpg",
        "nutrition": "High in vitamins, fatty acids, and protein; higher calories than chicken eggs.",
        "uses": "Gourmet dishes, baking, and cooking similar to chicken eggs."
    },
    "Asian quel": {
        "scientific_name": "Coturnix japonica",
        "color_size": "Color: Cream with brown splotches | Size: Very Small (1/3 chicken egg size)",
        "habitat": "Grasslands, farms, and forests in Asia",
        "fun_fact": "Quail eggs are a delicacy in Japan, often eaten raw in sushi.",
        "image": "images/quail_egg.jpg",
        "nutrition": "Rich in vitamins B12, selenium, riboflavin, choline, and iron; low calories.",
        "uses": "Sushi, appetizers, pickled, boiled, or in salads."
    },
    "quail egg": {
        "scientific_name": "Coturnix japonica",
        "color_size": "Color: Cream with brown splotches | Size: Very Small (1/3 chicken egg size)",
        "habitat": "Grasslands, farms, and forests worldwide",
        "fun_fact": "Quail eggs are considered a delicacy and are often eaten raw in some cultures.",
        "image": "images/quail_egg.jpg",
        "nutrition": "Rich in vitamins A and B, low in calories.",
        "uses": "Pickled, boiled, or used in appetizers."
    },
    "emu": {
        "scientific_name": "Dromaius novaehollandiae",
        "color_size": "Color: Dark green | Size: Large",
        "habitat": "Australian outback and farms",
        "fun_fact": "Emu eggs are dark green and can be carved as art pieces.",
        "image": "images/emu_egg.jpg",
        "nutrition": "Similar to ostrich, high in healthy fats.",
        "uses": "Eaten or used in indigenous Australian culture."
    },
    "black crow egg": {
        "scientific_name": "Corvus brachyrhynchos",
        "color_size": "Color: Greenish with brown/gray blotches | Size: Small (oval, ~40mm)",
        "habitat": "Urban areas, forests, fields across North America",
        "fun_fact": "Crows are highly social and use tools; eggs are rarely eaten by humans.",
        "image": "images/crow_egg.jpg",
        "nutrition": "Similar to other bird eggs: protein, fats; not commonly consumed.",
        "uses": "Scientific study; not typically culinary."
    },
    "turtle eggs": {
        "scientific_name": "Chelonia mydas (and other species)",
        "color_size": "Color: White/Leathery | Size: Varies (ping-pong ball size)",
        "habitat": "Oceans, beaches, and freshwater",
        "fun_fact": "Sea turtle eggs are protected in many countries due to endangered species status.",
        "image": "images/turtle_egg.jpg",
        "nutrition": "High in protein, but consumption is regulated.",
        "uses": "In some cultures, eaten as a delicacy, but often illegal."
    },
    "black drangos": {
        "scientific_name": "Dicrurus macrocercus",
        "color_size": "Color: White to pinkish with reddish-brown blotches | Size: Small (~24-27mm)",
        "habitat": "Savannas, fields, urban areas in tropical Asia",
        "fun_fact": "Black drongos mimic other birds to steal food and are bold predators.",
        "image": "images/black_drongo_egg.jpg",
        "nutrition": "Similar to other bird eggs; not commonly eaten.",
        "uses": "Not typically consumed; ecological role in insect control."
    },
    "blad eagle egg": {
        "scientific_name": "Haliaeetus leucocephalus",
        "color_size": "Color: Matte white | Size: Large (~74mm x 56mm)",
        "habitat": "Lakes, rivers, coastal areas near water in North America",
        "fun_fact": "Bald eagle eggs are protected; the bird is the U.S. national symbol.",
        "image": "images/blad_eagle_egg.jpg",
        "nutrition": "High in protein; consumption illegal due to protection.",
        "uses": "Protected; not for consumption."
    },
    "bulbul eggs": {
        "scientific_name": "Pycnonotus jocosus",
        "color_size": "Color: Pale pinkish with red spots | Size: Small (~20mm)",
        "habitat": "Gardens, scrub, forests in Indian subcontinent",
        "fun_fact": "Red-vented bulbuls are invasive in some areas and can't synthesize vitamin C.",
        "image": "images/bulbul_egg.jpg",
        "nutrition": "Similar to other bird eggs; not commonly eaten.",
        "uses": "Not typically culinary."
    },
    "cormorant": {
        "scientific_name": "Phalacrocorax carbo",
        "color_size": "Color: Chalky blue-white | Size: Medium (~50mm)",
        "habitat": "Coastal waters, lakes, rivers worldwide",
        "fun_fact": "Cormorants spread wings to dry feathers after diving; used in fishing.",
        "image": "images/cormorant_egg.jpeg",
        "nutrition": "High in protein from fish diet; not commonly eaten.",
        "uses": "Traditional fishing aid; eggs protected."
    },
    "crcodile": {
        "scientific_name": "Crocodylus niloticus (and other species)",
        "color_size": "Color: White, hard-shelled | Size: Varies (up to 7cm)",
        "habitat": "Rivers, swamps, wetlands in tropics",
        "fun_fact": "Crocodile eggs have temperature-dependent sex determination.",
        "image": "images/crocodile_egg.jpg",
        "nutrition": "High in protein; consumed in some cultures.",
        "uses": "Delicacy in some regions; regulated due to conservation."
    },
    "egdrt": {
        "scientific_name": "Egretta garzetta",
        "color_size": "Color: Greenish-blue | Size: Medium (~40mm)",
        "habitat": "Marshes, swamps, coastal areas worldwide",
        "fun_fact": "Egrets were nearly extinct for their plumes in the 1800s.",
        "image": "images/egret_egg.jpg",
        "nutrition": "Similar to other bird eggs; not commonly eaten.",
        "uses": "Protected; ecological indicator."
    },
    "hawk eggs": {
        "scientific_name": "Buteo jamaicensis",
        "color_size": "Color: White with brown blotches | Size: Medium (~50mm)",
        "habitat": "Open woodlands, fields, forests worldwide",
        "fun_fact": "Hawks have exceptional eyesight, spotting prey from afar.",
        "image": "images/hawk_egg.jpg",
        "nutrition": "High in protein; not commonly consumed.",
        "uses": "Falconry; protected."
    },
    "house sparrow": {
        "scientific_name": "Passer domesticus",
        "color_size": "Color: White, blue-white, or greenish-white with spots | Size: Small (~20mm)",
        "habitat": "Urban areas, farms, grasslands worldwide",
        "fun_fact": "House sparrows are highly adaptable and introduced globally.",
        "image": "images/house_sparrow_egg.jpg",
        "nutrition": "Similar to chicken eggs; good source of protein.",
        "uses": "Occasionally eaten; pet trade."
    },
    "king cobra egg": {
        "scientific_name": "Ophiophagus hannah",
        "color_size": "Color: White, leathery | Size: Large (~50mm)",
        "habitat": "Forests, bamboo thickets in Southeast Asia",
        "fun_fact": "King cobras build nests and guard eggs, unique among snakes.",
        "image": "images/king_cobra_egg.jpg",
        "nutrition": "High in protein; consumed in some cultures.",
        "uses": "Traditional medicine; delicacy (regulated)."
    },
    "kingfisher": {
        "scientific_name": "Alcedo atthis",
        "color_size": "Color: Glossy white | Size: Small (~30mm)",
        "habitat": "Rivers, streams, wetlands worldwide",
        "fun_fact": "Kingfishers dive at 25 mph to catch fish.",
        "image": "images/kingfisher_egg.jpeg",
        "nutrition": "Similar to other bird eggs; not commonly eaten.",
        "uses": "Ecological indicator; protected."
    },
    "lizard egg": {
        "scientific_name": "Agama sinaita (and other species)",
        "color_size": "Color: White | Size: Small (~1-2cm)",
        "habitat": "Varied: deserts, forests, urban areas worldwide",
        "fun_fact": "Some lizards lay eggs that hatch into live young (ovoviviparous).",
        "image": "images/lizard_egg.jpg",
        "nutrition": "High in protein; consumed in some cultures.",
        "uses": "Delicacy in Asia; pet trade."
    },
    "myna Egg": {
        "scientific_name": "Acridotheres tristis",
        "color_size": "Color: Blue-green | Size: Small (~30mm)",
        "habitat": "Urban areas, farmlands in Asia",
        "fun_fact": "Mynas are invasive and mimic sounds excellently.",
        "image": "images/myna_egg.jpg",
        "nutrition": "Similar to other bird eggs; not commonly eaten.",
        "uses": "Pest control (invasive); pets."
    },
    "owl image": {
        "scientific_name": "Tyto alba",
        "color_size": "Color: White | Size: Medium (~40mm)",
        "habitat": "Forests, urban areas worldwide",
        "fun_fact": "Owls can rotate heads 270 degrees.",
        "image": "images/owl_egg.jpg",
        "nutrition": "High in protein; not commonly consumed.",
        "uses": "Protected; symbolic in culture."
    },
    "parrot egg": {
        "scientific_name": "Psittacidae (family, e.g., Ara macao)",
        "color_size": "Color: White | Size: Medium (~35mm)",
        "habitat": "Tropical forests, savannas worldwide",
        "fun_fact": "Parrots are intelligent and mimic speech.",
        "image": "images/parrot_egg.webp",
        "nutrition": "Rich in vitamins; consumed in some regions.",
        "uses": "Pets, conservation focus."
    },
    "peacock egg": {
        "scientific_name": "Pavo cristatus",
        "color_size": "Color: White | Size: Large (~70mm)",
        "habitat": "Forests, farms in India and Southeast Asia",
        "fun_fact": "Peacock eggs are laid in ground nests; males display for mates.",
        "image": "images/peacock_egg.jpg",
        "nutrition": "High protein, vitamins; similar to chicken but larger.",
        "uses": "Gourmet, ornamental birds."
    },
    "piageon egg": {
        "scientific_name": "Columba livia",
        "color_size": "Color: White | Size: Small (~30mm)",
        "habitat": "Urban areas, farmlands worldwide",
        "fun_fact": "Pigeons can recognize themselves in mirrors.",
        "image": "images/pigeon_egg.jpg",
        "nutrition": "High protein, fats; richer yolk than chicken eggs.",
        "uses": "Squab meat, racing, messengers."
    },
    "stork bird": {
        "scientific_name": "Ciconia ciconia",
        "color_size": "Color: Chalky white | Size: Large (~70mm)",
        "habitat": "Wetlands, farmlands in Europe and Africa",
        "fun_fact": "Storks are symbols of birth; migrate long distances.",
        "image": "images/stork_egg.jpg",
        "nutrition": "High in protein; not commonly eaten.",
        "uses": "Cultural symbol; protected."
    }
}

# Expanded predefined chatbot responses
chat_responses = {
    "which animals can this model detect?": "It can detect eggs from chicken, duck, ostrich, turkey, Asian quel, quail egg, emu, black crow egg, turtle eggs, black drangos, blad eagle egg, bulbul eggs, cormorant, crcodile, egdrt, hawk eggs, house sparrow, king cobra egg, kingfisher, lizard egg, myna Egg, owl image, parrot egg, peacock egg, piageon egg, and stork bird.",
    "how accurate is it?": "The model has an accuracy of around 68-70% on validation data, but real-world performance may vary based on image quality.",
    "can it detect fake eggs?": "No, this model focuses on species classification based on visual features like shape, color, and texture. It doesn't detect synthetic or fake eggs.",
    "what input size is required?": "Images are automatically resized to 224x224 pixels for the model, but higher resolution uploads yield better results.",
    "tell me more about chicken eggs": "Chicken eggs (Gallus gallus domesticus) are versatile, nutritious, and come in various sizes. Fun fact: The color of the shell depends on the chicken's breed!",
    "what's the largest egg?": "The ostrich egg (Struthio camelus) is the largest, weighing up to 1.5 kg!",
    "are turtle eggs edible?": "Turtle eggs (e.g., Chelonia mydas) are edible in some cultures, but many turtle species are endangered, so consumption is often regulated or illegal.",
    "how does the model work?": "It uses a convolutional neural network (CNN) trained on thousands of egg images to classify based on features like color, size, and patterns.",
    "can i train it further?": "Yes, but that requires additional data and TensorFlow expertise. Contact the developer for more info.",
    "what's a fun fact about quail eggs?": "Quail eggs (Coturnix japonica) are tiny but pack a nutritional punch and are often used in fancy appetizers."
}

@app.route('/')
def home():
    return render_template('index.html', egg_types=list(egg_info.keys()), egg_info=egg_info)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocess image
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    preds = model.predict(img_array)
    pred_index = np.argmax(preds)
    confidence = round(np.max(preds) * 100, 2)
    label = labels.get(pred_index, f"Class_{pred_index}")

    # Get egg info dictionary
    info = egg_info.get(label.lower(), None)

    # Get top 3 predictions for more detail
    top_indices = np.argsort(preds[0])[-3:][::-1]
    top_preds = [(labels.get(i, f"Class_{i}"), round(preds[0][i] * 100, 2)) for i in top_indices]

    return render_template('result.html',
                           label=label,
                           confidence=confidence,
                           img_path=file_path.replace('\\', '/'),
                           info=info,
                           top_preds=top_preds)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    question = request.form.get("question", "").lower()
    # Fuzzy matching for better interactivity
    for key in chat_responses:
        if key in question:
            return chat_responses[key]
    return "Sorry, I can't answer that right now. Try asking about specific eggs or model details!"

@app.route('/encyclopedia/<egg_type>')
def encyclopedia(egg_type):
    info = egg_info.get(egg_type.lower(), None)
    if not info:
        return "Egg type not found", 404
    return render_template('encyclopedia.html', info=info, egg_type=egg_type)

if __name__ == '__main__':
    app.run(debug=True)