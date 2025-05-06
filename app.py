from flask import Flask, render_template, request, redirect, url_for, session
import json

app = Flask(__name__)
app.secret_key = 'tajne_kluc_slova'

# Načíta strom z JSON
with open("tree.json", encoding="utf-8") as f:
    tree = json.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    if "node" not in session:
        session["node"] = tree  # začni od koreňa

    node = session["node"]

    # Ak už máme výsledok
    if "result" in node:
        result = node["result"]
        session.clear()
        return render_template("result.html", result=result)

    if request.method == "POST":
        if node.get("type", "binary") == "numeric":
            try:
                user_value = float(request.form.get("answer"))
                threshold = node["threshold"]
                answer = "yes" if user_value > threshold else "no"
            except (ValueError, TypeError):
                return "Zadajte platné číslo", 400
        else:
            answer = request.form.get("answer")

        if answer in node["answers"]:
            session["node"] = node["answers"][answer]
            return redirect(url_for("index"))
        else:
            return "Neplatná odpoveď", 400

    return render_template("question.html", question=node["question"], type=node.get("type", "binary"))


if __name__ == "__main__":
    app.run(debug=True)
