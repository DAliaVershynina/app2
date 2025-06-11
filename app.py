from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

# Inicializácia aplikácie
from flask import Flask, render_template
import os

template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
app = Flask(__name__, template_folder=template_dir)


# Načítanie modelu (predtým uloženého)
model = joblib.load("best_rf_model.pkl")  # Nahraď cestu k svojmu modelu
ATTRIBUTES = [
    "B1", "B2", "B3", "B4", "C1", "C2", "C3",
    "D1", "D2", "D3", "D4", "D5", "D6",
    "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10",
    "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9",
    "G",
    "H1", "H2", "H3", "H4", "H5", "H6", "H8", "H9", "H10", "H11", "H12", "H14",
    "I1", "I2", "I3", "I4",
    "J1", "J2", "J3",
    "K1", "K2", "K3", "K4",
    "L", "M",
    "N1", "N2", "N3", "N4", "N5", "N6", "N7",
    "O1", "O2", "O4", "O5",
    "P1", "P3", "P4", "P5", "P6", "P7", "P9", "P10", "P11", "P13", "P14", "P15",
    "P16", "P17", "P18", "P20", "P21", "P23", "P24", "P25", "P26", "P27", "P28",
    "P29", "P31", "P32", "P33",
    "R1", "R2", "R3"
]
# Všetky atribúty, ktoré očakáva model (v rovnakom poradí ako pri trénovaní)
ATTRIBUTES_LABELS = {
    "B1": "Som vyšetrovaný pre Stratu vedomia",
    "B2": 'Som vyšetrovaný pre Pocity hroziacej straty vedomia',
    "B3":'Som vyšetrovaný pre Stav po resuscitácii',
    "B4":'Som vyšetrovaný pre Stav po eplileptickom záchvate',
    "C1":'Začiatok ťažkostí (približný dátum)?',
    "C2":'Počet odpadnutí spolu?',
    "C3":'Kedy bolo posledné odpadnutie?',
    "D1":'Popíšte ťažkosti / V akej situácii vznikli Strata vedomia pri státí?',
    "D2":'Popíšte ťažkosti / V akej situácii vznikli Strata vedomia do 1 minúty po postavení sa?',
    "D3":'Popíšte ťažkosti / V akej situácii vznikli Pri chôdzi?',
    "D4":'Popíšte ťažkosti / V akej situácii vznikli Pri fyzickej námahe (akej?)',
    "D5":'Popíšte ťažkosti / V akej situácii vznikli Strata vedomia v sede?',
    "D6":'Popíšte ťažkosti / V akej situácii vznikli Strata vedomia poležačky?',
    "E1":'Čo viedlo k strate vedomia Preľudnené priestory?',
    "E2": 'Čo viedlo k strate vedomia Dusné prostredie?',
    "E3": 'Čo viedlo k strate vedomia Teplé prostredie?',
    "E4": 'Čo viedlo k strate vedomia Pohľad na krv?',
    "E5": 'Čo viedlo k strate vedomia Nepríjemné emócie (strach, úzkosť, rozrušenie, odpor, pohľad na násilie)',
    "E6": 'Čo viedlo k strate vedomia Medicínsky výkon?',
    "E7": 'Čo viedlo k strate vedomia Bolesť?',
    "E8": 'Čo viedlo k strate vedomia Dehydratácia?',
    "E9": 'Čo viedlo k strate vedomia Menštruácia?',
    "E10": 'Čo viedlo k strate vedomia Strata krvi?',
    "F1": 'Vznikla strata vedomia pri niektorej z týchto situácií? Pri stolici',
    "F2": 'Vznikla strata vedomia pri niektorej z týchto situácií? Pri močení',
    "F3": 'Vznikla strata vedomia pri niektorej z týchto situácií? Pri kašli',
    "F4": 'Vznikla strata vedomia pri niektorej z týchto situácií? Pri kýchaní/smrkaní nosa',
    "F5": 'Vznikla strata vedomia pri niektorej z týchto situácií? Pri jedení/prehĺtaní',
    "F6": 'Vznikla strata vedomia pri niektorej z týchto situácií? Po náhlej bolesti',
    "F7": 'Vznikla strata vedomia pri niektorej z týchto situácií? Počas fyzickej námahy',
    "F8": 'Vznikla strata vedomia pri niektorej z týchto situácií? Pri hlade',
    "F9": 'Vznikla strata vedomia pri niektorej z týchto situácií? Pri nedostatku spánku, únave',
    "G": 'Užili ste hodinu pred stratou vedomia nejaké lieky alebo alkohol?',
    "H1": 'Čo ste cítili tesne pred stratou vedomia? Pocit na zvracanie alebo zvracanie',
    "H2": 'Čo ste cítili tesne pred stratou vedomia? Pocit tepla/horúco',
    "H3": 'Čo ste cítili tesne pred stratou vedomia? Pot',
    "H4": 'Čo ste cítili tesne pred stratou vedomia? Zahmlievanie pred očami',
    "H5": 'Čo ste cítili tesne pred stratou vedomia? Hučanie v ušiach',
    "H6": 'Čo ste cítili tesne pred stratou vedomia? Búšenie srdca1',
    "H8": 'Čo ste cítili tesne pred stratou vedomia? Bolesť na hrudníku',
    "H9": 'Čo ste cítili tesne pred stratou vedomia? Neobvyklý zápach',
    "H10": 'Čo ste cítili tesne pred stratou vedomia? Neobvyklé zvuky',
    "H11": 'Čo ste cítili tesne pred stratou vedomia? Poruchy reči alebo slabosť polovice tela',
    "H12": 'Čo ste cítili tesne pred stratou vedomia? Nepociťoval som nič zvláštne',
    "H14": 'Čo ste cítili tesne pred stratou vedomia? Iné',
    "I1": 'Ako dlho trvali tieto pocity pred stratou vedomia? Niekoľko sekúnd1',
    "I2": 'Ako dlho trvali tieto pocity pred stratou vedomia? Do 1 minúty',
    "I3": 'Ako dlho trvali tieto pocity pred stratou vedomia? Do 5 minút',
    "I4": 'Ako dlho trvali tieto pocity pred stratou vedomia? Viac ako 5 minút',
    "J1": 'Čo ste urobili pri hroziacej strate vedomia? Sadol som si',
    "J2": 'Čo ste urobili pri hroziacej strate vedomia? Ľahol som si',
    "J3": 'Čo ste urobili pri hroziacej strate vedomia? Nestihol som urobiť nič pretože som stratil vedomie',
    "K1": 'Ak boli prítomní svedkovia, ako dlho podľa nich trvalo bezvedomie? Niekoľko sekúnd',
    "K2": 'Ak boli prítomní svedkovia, ako dlho podľa nich trvalo bezvedomie? Do minúty',
    "K3": 'Ak boli prítomní svedkovia, ako dlho podľa nich trvalo bezvedomie? Do 5 minút',
    "K4": 'Ak boli prítomní svedkovia, ako dlho podľa nich trvalo bezvedomie? Viac ako 5 minút',
    "L": 'Mali ste kŕče počas bezvedomia (v prípade svedkov udalosti)?',
    "M": 'Odišla vám stolica alebo moč počas bezvedomia?',
    "N1": 'Pamätáte si na udalosti po strate vedomia? Mali ste pohryzený jazyk, pery?',
    "N2": 'Pamätáte si na udalosti po strate vedomia? Udreli ste sa pri páde, boli ste poranení v dôsledku pádu?',
    "N3": 'Pamätáte si na udalosti po strate vedomia? Po prebratí ste podľa údajov svedkov boli viac ako 30 minút dozerientovaní?',
    "N4": 'Pamätáte si na udalosti po strate vedomia? Bolela vás hlava alebo svaly?',
    "N5": 'Pamätáte si na udalosti po strate vedomia? Aj po prebratí ste pociťovali nevoľnosť?',
    "N6": 'Pamätáte si na udalosti po strate vedomia? Cítili ste sa normálne',
    "N7": 'Pamätáte si na udalosti po strate vedomia? Nepamätáte sa',
    "O1": 'Výskyt ochorení vo vašej rodine Náhle úmrtie člena rodiny (v akom veku?)',
    "O2": 'Výskyt ochorení vo vašej rodine Ochorenie srdca',
    "O4": 'Výskyt ochorení vo vašej rodine Srdcová arytmia/kardiostimulátor',
    "O5": 'Výskyt ochorení vo vašej rodine Ochoria mozgu/epilepsia',
    "P1": 'Na aké ochorenia ste sa doteraz liečili? Ochorenie srdca',
    "P3": 'Na aké ochorenia ste sa doteraz liečili? Ochorenie chlopní',
    "P4": 'Na aké ochorenia ste sa doteraz liečili? Srdcová slabosť',
    "P5": 'Na aké ochorenia ste sa doteraz liečili? Koronárna chorova srdca',
    "P6": 'Na aké ochorenia ste sa doteraz liečili? Srdcové arytmie',
    "P7": 'Na aké ochorenia ste sa doteraz liečili? Búšenie srdca',
    "P9": 'Na aké ochorenia ste sa doteraz liečili? Bolesti na hrudníku',
    "P10": 'Na aké ochorenia ste sa doteraz liečili? Vysoký tlak krvi',
    "P11": 'Na aké ochorenia ste sa doteraz liečili? Nízky tlak krvi',
    "P13": 'Na aké ochorenia ste sa doteraz liečili? Ochorenia obličiek',
    "P14": 'Na aké ochorenia ste sa doteraz liečili? Diabetes (cukrovka)',
    "P15": 'Na aké ochorenia ste sa doteraz liečili? Anémia',
    "P16": 'Na aké ochorenia ste sa doteraz liečili? Astma',
    "P17": 'Na aké ochorenia ste sa doteraz liečili? Ochorenia pľúc',
    "P18": 'Na aké ochorenia ste sa doteraz liečili? Ochorenia priedušiek',
    "P20": 'Na aké ochorenia ste sa doteraz liečili? Ochorenia čreva',
    "P21": 'Na aké ochorenia ste sa doteraz liečili? Ochorenia štítnej žľazy',
    "P23": 'Na aké ochorenia ste sa doteraz liečili? Bolesti hlavy',
    "P24": 'Na aké ochorenia ste sa doteraz liečili? Neurologické ochorenia',
    "P25": 'Na aké ochorenia ste sa doteraz liečili? Parkinsonová choroba',
    "P26": 'Na aké ochorenia ste sa doteraz liečili? Psychiatrické ochorenia ',
    "P27": 'Na aké ochorenia ste sa doteraz liečili? Depresia',
    "P28": 'Na aké ochorenia ste sa doteraz liečili? Ochorenia krčnej chrbtice',
    "P29": 'Na aké ochorenia ste sa doteraz liečili? Bolesti chrbta',
    "P31": 'Na aké ochorenia ste sa doteraz liečili? Nádorové ochorenie',
    "P32": 'Na aké ochorenia ste sa doteraz liečili? Prekonané operácie',
    "P33": 'Na aké ochorenia ste sa doteraz liečili? Prekonané úrazy',
    "R1": 'Boli ste v poslednom období očkovaní (cca za posledných 10-15 rokov)? Proti HPV (rakovina krčka maternice)',
    "R2": 'Boli ste v poslednom období očkovaní (cca za posledných 10-15 rokov)? Proti chrípke',
    "R3": 'Boli ste v poslednom období očkovaní (cca za posledných 10-15 rokov)? Iné'

}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        input_data = []
        for attr in ATTRIBUTES:
            value = request.form.get(attr, -1)
            try:
                input_data.append(float(value))
            except ValueError:
                input_data.append(-1)

        user_df = pd.DataFrame([input_data], columns=ATTRIBUTES)
        probs = model.predict_proba(user_df)[0]
        prob_neg = round(probs[0] * 100, 2)
        prob_pos = round(probs[1] * 100, 2)

        result = {
            "positive": prob_pos,
            "negative": prob_neg,
            "interpretation": "Pozitívny HUT test je pravdepodobný." if prob_pos >= 50 else "Negatívny HUT test je pravdepodobný."
        }

    return render_template("index.html", attributes=ATTRIBUTES, labels=ATTRIBUTES_LABELS, result=result)

if __name__ == "__main__":
    app.run(debug=True)
