from transformers import AutoTokenizer, AutoModelForCausalLM

class CodeEvaluator:
    def __init__(self):
        # Load a larger model like StarCoder or GPT-4
        self.tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder")  # Or "codellama/CodeLlama-7b-hf"
        self.model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder")  # Or "codellama/CodeLlama-7b-hf"

    def evaluate_criterion(self, ref_code, answer_code, criterion, max_score):
        """
        Evaluate a single criterion using a large generative AI model.
        Returns a score and feedback.
        """
        prompt = (
            f"Analyze the following Java code and evaluate the criterion: {criterion}\n\n"
            f"Reference Code:\n{ref_code}\n\n"
            f"Student Code:\n{answer_code}\n\n"
            f"Assign a score out of {max_score} based on how well the criterion is met.\n"
            f"If the criterion is fully met, assign the maximum score.\n"
            f"If the criterion is partially met, assign a score proportional to its correctness.\n"
            f"If the criterion is not met at all, assign a score of 0.\n"
            f"Provide detailed feedback explaining why this score was assigned."
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model.generate(**inputs, max_length=512)
        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse the output
        try:
            score_str = raw_output.split("Score:")[-1].strip().split()[0]
            score = int(score_str)
        except:
            score = 0  # Default to 0 if score extraction fails

        feedback = raw_output.strip()
        return score, feedback

    def evaluate_submission(self, ref_code, answer_code, rubric):
        """
        Evaluate the student's submission based on the dynamic rubric.
        """
        result = {
            "total_score": 0,
            "grades": {},
            "feedback": {}
        }

        # Evaluate each criterion in the rubric
        for criterion, max_score in rubric.items():
            score, feedback = self.evaluate_criterion(ref_code, answer_code, criterion, max_score)
            result["grades"][criterion] = score
            result["feedback"][criterion] = feedback
            result["total_score"] += score

        return result