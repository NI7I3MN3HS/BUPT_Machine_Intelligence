from flask import Flask, render_template, request
from A_star import A_star

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/solve", methods=["POST"])
def solve():
    start_state = request.form["start_state"]
    target_state = request.form["target_state"]

    is_solve, steps = A_star(start_state, target_state)
    if is_solve != 0:
        result = "无解"
    else:
        result = []
        for i in range(len(steps)):
            result.append(
                {"step": i + 1, "state": [steps[i][:3], steps[i][3:6], steps[i][6:]]}
            )

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
