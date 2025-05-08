from flask import Flask, render_template, request, redirect, url_for, session
import json

app = Flask(__name__)
app.secret_key = 'tajne_kluc_slova'

# Načítanie stromu rozhodovania zo súboru
with open("tree.json", encoding="utf-8") as f:
    tree = json.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    if "node" not in session:
        session["node"] = tree  # Začiatok v koreni stromu
        session["answers"] = []  # Tu uchovávame odpovede používateľa

    node = session["node"]

    # Ak už máme výstup – zobrazíme výsledok + rekapituláciu
    if "result" in node:
        result = node["result"]
        confidence = node.get("confidence", "–")
        answers = session.get("answers", [])
        session.clear()
        return render_template("result.html", result=result, answers=answers, confidence=confidence)

    # Spracovanie odpovede
    if request.method == "POST":
        if node.get("type", "binary") == "numeric":
            try:
                user_value = float(request.form.get("numeric_answer"))
                threshold = node["threshold"]
                direction = "yes" if user_value > threshold else "no"
                answer_label = f"{node['feature']} = {user_value}"
            except (ValueError, TypeError):
                return "Zadajte platné číslo", 400
        else:
            direction = request.form.get("answer")
            answer_label = f"{node['feature']} = {'Áno' if direction == '1' else 'Nie'}"

        # Uloženie odpovede pre rekapituláciu
        session["answers"].append({"question": node["question"], "odpoved": answer_label})

        if direction in node["answers"]:
            session["node"] = node["answers"][direction]
            return redirect(url_for("index"))
        else:
            return "Neplatná odpoveď", 400

    return render_template(
        "question.html",
        question=node["question"],
        type=node.get("type", "binary")
    )

if __name__ == "__main__":
    app.run(debug=True)
