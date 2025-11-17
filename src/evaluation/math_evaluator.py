# src/evaluation/math_evaluator.py
import sympy as sp
from sympy import symbols, simplify, solve, Eq
import re

class MathEvaluator:
    def __init__(self):
        self.x, self.y = symbols('x y')
    
    def clean_expression(self, expression_str):
        """Clean and normalize the expression string"""
        # Replace common handwritten symbols
        expression_str = expression_str.replace('×', '*').replace('÷', '/')
        expression_str = expression_str.replace(' ', '')  # Remove spaces
        
        # Handle equations (split on = if present)
        if '=' in expression_str:
            parts = expression_str.split('=')
            if len(parts) == 2:
                left, right = parts
                # For equations, we'll compare both sides
                return f"Eq({left}, {right})"
        
        return expression_str
    
    def parse_expression(self, expression_str):
        """Parse math expression from string"""
        try:
            # Clean the expression first
            cleaned_expr = self.clean_expression(expression_str)
            
            # Handle equations
            if cleaned_expr.startswith('Eq('):
                # Extract the equation parts
                match = re.match(r'Eq\((.*),(.*)\)', cleaned_expr)
                if match:
                    left = sp.sympify(match.group(1).strip())
                    right = sp.sympify(match.group(2).strip())
                    return Eq(left, right)
            
            # Parse regular expressions
            expr = sp.sympify(cleaned_expr)
            return expr
            
        except Exception as e:
            print(f"Error parsing expression '{expression_str}': {e}")
            return None
    
    def compare_expressions(self, student_expr, correct_expr):
        """Compare student answer with correct solution"""
        try:
            # Parse both expressions
            student_parsed = self.parse_expression(student_expr)
            correct_parsed = self.parse_expression(correct_expr)
            
            if student_parsed is None or correct_parsed is None:
                return False, "Parsing error - check expression format"
            
            # Handle equations
            if isinstance(student_parsed, Eq) and isinstance(correct_parsed, Eq):
                # For equations, check if they're equivalent
                student_simplified = simplify(student_parsed.lhs - student_parsed.rhs)
                correct_simplified = simplify(correct_parsed.lhs - correct_parsed.rhs)
                are_equivalent = student_simplified.equals(correct_simplified)
                message = "Equations are equivalent" if are_equivalent else "Equations differ"
                return are_equivalent, message
            
            # Handle regular expressions
            elif not isinstance(student_parsed, Eq) and not isinstance(correct_parsed, Eq):
                student_simple = simplify(student_parsed)
                correct_simple = simplify(correct_parsed)
                are_equivalent = student_simple.equals(correct_simple)
                message = "Expressions are equivalent" if are_equivalent else "Expressions differ"
                return are_equivalent, message
            
            else:
                return False, "Cannot compare equation with expression"
                
        except Exception as e:
            return False, f"Comparison error: {str(e)}"
    
    def compare_numerical(self, student_answer, correct_answer, tolerance=0.001):
        """Compare numerical answers with tolerance"""
        try:
            student_val = float(student_answer)
            correct_val = float(correct_answer)
            return abs(student_val - correct_val) < tolerance
        except:
            return False
    
    def preprocess_ocr_errors(self, text):
        """Fix common OCR errors before comparison"""
        # Handle equals sign confusion
        text = text.replace('−', '=').replace('–', '=').replace('—', '=')
        
        # Common pattern: 2+2-4 should be 2+2=4
        patterns = [
            (r'(\d+)\+(\d+)\-(\d+)', r'\1+\2=\3'),  # 2+2-4 → 2+2=4
            (r'(\d+)\-(\d+)\=(\d+)', r'\1-\2=\3'),   # Fix minus-equals
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def calculate_score(self, student_answer, correct_answer, problem_type="auto"):
        """Calculate grading score with OCR error tolerance"""
        # Pre-process both answers to handle common OCR issues
        student_cleaned = self.preprocess_ocr_errors(student_answer)
        correct_cleaned = self.preprocess_ocr_errors(correct_answer)
        
        print(f"🔧 Comparing: '{student_cleaned}' vs '{correct_cleaned}'")
        
        # Auto-detect problem type
        if problem_type == "auto":
            if '=' in student_cleaned or '=' in correct_cleaned:
                problem_type = "equation"
            else:
                # Check if it's likely numerical
                try:
                    float(student_cleaned)
                    float(correct_cleaned)
                    problem_type = "numerical"
                except:
                    problem_type = "expression"
        
        if problem_type == "numerical":
            is_correct = self.compare_numerical(student_cleaned, correct_cleaned)
            score = 1.0 if is_correct else 0.0
            message = "Correct numerical answer" if is_correct else "Incorrect numerical answer"
        
        elif problem_type == "equation":
            is_correct, message = self.compare_expressions(student_cleaned, correct_cleaned)
            score = 1.0 if is_correct else 0.0
        
        else:  # expression
            is_correct, message = self.compare_expressions(student_cleaned, correct_cleaned)
            score = 1.0 if is_correct else 0.0
        
        return {
            'score': score,
            'is_correct': is_correct,
            'feedback': message,
            'student_answer': student_cleaned,
            'correct_answer': correct_cleaned,
            'problem_type': problem_type
        }