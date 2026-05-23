from sentence_transformers import SentenceTransformer, util
import numpy as np

bert_model = SentenceTransformer('all-MiniLM-L6-v2')

intent_templates = {

    # ── Soil / Nitrogen level ──────────────────────────────────────────
    'soil_low_nitrogen': [
        "soil with very low nitrogen content",
        "nitrogen deficient ground poor in N",
        "sandy or light soil lacking nutrients",
        "infertile soil low in nitrogen",
    ],
    'soil_medium_nitrogen': [
        "soil with moderate nitrogen level",
        "average nitrogen content in the field",
        "balanced soil nutrients medium N",
    ],
    'soil_high_nitrogen': [
        "soil rich in nitrogen very fertile",
        "high nitrogen content dark rich earth",
        "nitrogen-rich fertile loamy field",
    ],

    # ── Temperature ───────────────────────────────────────────────────
    'temp_very_cold': [
        "very cold weather below 15 degrees",
        "freezing temperatures frost conditions",
        "extremely cold climate icy",
    ],
    'temp_cool': [
        "cool mild weather around 18 to 22 degrees",
        "moderate cold temperate conditions",
        "cool climate spring-like weather",
    ],
    'temp_warm': [
        "warm weather between 23 and 28 degrees",
        "pleasant warm conditions subtropical",
        "mild warm tropical weather",
    ],
    'temp_hot': [
        "hot weather above 30 degrees",
        "very hot scorching heat summer",
        "high temperature arid hot climate",
    ],
    'temp_very_hot': [
        "extremely hot above 35 degrees",
        "scorching extreme heat desert conditions",
        "very high temperature dry heat",
    ],

    # ── Humidity ──────────────────────────────────────────────────────
    'humidity_very_low': [
        "very dry air extremely low humidity",
        "arid dry conditions almost no moisture",
        "desert-like very low humidity",
    ],
    'humidity_low': [
        "low humidity dry air conditions",
        "semi-arid dry moderate humidity",
        "relatively dry air low moisture",
    ],
    'humidity_moderate': [
        "moderate humidity balanced moisture",
        "average humidity comfortable conditions",
        "normal moisture levels in air",
    ],
    'humidity_high': [
        "high humidity moist humid conditions",
        "very humid tropical moist air",
        "muggy damp high moisture environment",
    ],
    'humidity_very_high': [
        "extremely high humidity almost saturated",
        "very moist tropical rainforest conditions",
        "near 100 percent humidity waterlogged air",
    ],

    # ── Rainfall ──────────────────────────────────────────────────────
    'rainfall_very_low': [
        "very little rain almost no rainfall",
        "drought conditions extremely scarce precipitation",
        "barely any rain very dry season",
    ],
    'rainfall_low': [
        "low rainfall below average precipitation",
        "not much rain semi-arid conditions",
        "sparse rainfall dry season",
    ],
    'rainfall_moderate': [
        "moderate rainfall average precipitation",
        "normal rain levels balanced moisture",
        "regular seasonal rainfall",
    ],
    'rainfall_high': [
        "high rainfall heavy rains",
        "lots of rain heavy precipitation",
        "rainy season heavy showers frequent rain",
    ],
    'rainfall_very_high': [
        "extremely heavy rainfall flooding",
        "monsoon heavy downpours very high precipitation",
        "near 300mm rainfall tropical monsoon",
    ],

    # ── Soil pH ───────────────────────────────────────────────────────
    'ph_acidic': [
        "acidic soil low pH below 6",
        "highly acidic ground sour soil",
        "acid soil pH around 5 or less",
    ],
    'ph_slightly_acidic': [
        "slightly acidic soil pH around 6",
        "mildly acidic soil common farming land",
        "near neutral slightly acidic ground",
    ],
    'ph_neutral': [
        "neutral soil pH around 6.5 to 7",
        "balanced neutral soil neither acidic nor alkaline",
        "ideal pH neutral fertile soil",
    ],
    'ph_alkaline': [
        "alkaline soil high pH above 7",
        "basic alkaline ground high pH soil",
        "soil pH above 7.5 alkaline conditions",
    ],
}

encoded_templates = {}

for intent, phrases in intent_templates.items():
    encoded_templates[intent] = bert_model.encode(phrases, convert_to_tensor=True)


def extract_features_bert(query, threshold=0.30):
    query_embedding = bert_model.encode(query, convert_to_tensor=True)

    scores = {}

    # compute similarity for all intents
    for intent, template_embeddings in encoded_templates.items():
        sims = util.cos_sim(query_embedding, template_embeddings)
        scores[intent] = sims.max().item()

    categories = {
        'nitrogen':    ['soil_low_nitrogen', 'soil_medium_nitrogen', 'soil_high_nitrogen'],
        'temperature': ['temp_very_cold', 'temp_cool', 'temp_warm', 'temp_hot', 'temp_very_hot'],
        'humidity':    ['humidity_very_low', 'humidity_low', 'humidity_moderate',
                        'humidity_high', 'humidity_very_high'],
        'rainfall':    ['rainfall_very_low', 'rainfall_low', 'rainfall_moderate',
                        'rainfall_high', 'rainfall_very_high'],
        'ph':          ['ph_acidic', 'ph_slightly_acidic', 'ph_neutral', 'ph_alkaline'],
    }

    extracted = {}

    for category, intents in categories.items():
        best_intent = max(intents, key=lambda i: scores[i])
        extracted[category] = best_intent if scores[best_intent] >= threshold else None

    return extracted, scores

def generate_feature_vector(extracted):

    N = 50.6
    P = 53.4
    K = 48.1
    temperature = 25.6
    humidity = 71.5
    ph = 6.47
    rainfall = 103.5

    NITROGEN_MAP = {
        'soil_low_nitrogen': 20,
        'soil_medium_nitrogen': 50,
        'soil_high_nitrogen': 100,
    }

    TEMPERATURE_MAP = {
        'temp_very_cold': 12.0,
        'temp_cool': 20.0,
        'temp_warm': 25.5,
        'temp_hot': 31.0,
        'temp_very_hot': 38.0,
    }

    HUMIDITY_MAP = {
        'humidity_very_low': 18.0,
        'humidity_low': 52.0,
        'humidity_moderate': 65.0,
        'humidity_high': 82.0,
        'humidity_very_high': 93.0,
    }

    RAINFALL_MAP = {
        'rainfall_very_low': 28.0,
        'rainfall_low': 52.0,
        'rainfall_moderate': 94.0,
        'rainfall_high': 160.0,
        'rainfall_very_high': 240.0,
    }

    PH_MAP = {
        'ph_acidic': 5.0,
        'ph_slightly_acidic': 6.0,
        'ph_neutral': 6.5,
        'ph_alkaline': 7.5,
    }

    if extracted.get('nitrogen'):
        N = NITROGEN_MAP.get(extracted['nitrogen'], N)

    if extracted.get('temperature'):
        temperature = TEMPERATURE_MAP.get(extracted['temperature'], temperature)

    if extracted.get('humidity'):
        humidity = HUMIDITY_MAP.get(extracted['humidity'], humidity)

    if extracted.get('rainfall'):
        rainfall = RAINFALL_MAP.get(extracted['rainfall'], rainfall)

    if extracted.get('ph'):
        ph = PH_MAP.get(extracted['ph'], ph)

    return [[N, P, K, temperature, humidity, ph, rainfall]]