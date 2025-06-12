import sys
import os
# Add path to loma_public
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../loma_public")))

import compiler


def generate_differentiated_matrix():
    with open("matrix.py") as f:
        code = f.read()

    differentiated_funcs = [
        "matrix_multiply",
        "matrix_vector_multiply",
        "vector_matrix_multiply"
    ]

    code += "\n"
    for func in differentiated_funcs:
        code += f"{func}_diff = ReverseDiff('{func}_diff', '{func}')\n"

    structs, lib = compiler.compile(code,
                                    target='c',
                                    output_filename="_code/matrix_diff")

    return structs, lib

if __name__ == "__main__":
    generate_differentiated_matrix()