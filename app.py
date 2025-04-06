from flask import Flask, request, jsonify
from model import CodeEvaluator

app = Flask(__name__)
evaluator = CodeEvaluator()

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    ref_code = data.get("reference_code")
    answer_code = data.get("answer_code")
    rubric = data.get("rubric")

    if not ref_code or not answer_code or not rubric:
        return jsonify({"status": "error", "message": "Missing required fields"}), 400

    result = evaluator.evaluate_submission(ref_code, answer_code, rubric)
    return jsonify({"status": "success", "result": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)