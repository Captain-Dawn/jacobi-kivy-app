from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image
import numpy as np

MAX_ITER = 50


# ---------- MATRIX UTILITIES ----------

def is_diagonally_dominant(A):
    for i in range(len(A)):
        diag = abs(A[i][i])
        others = sum(abs(A[i][j]) for j in range(len(A)) if j != i)
        if diag < others:
            return False
    return True


def rearrange_to_diagonal_dominance(A, b):
    n = len(A)
    A = A.copy()
    b = b.copy()

    for i in range(n):
        max_row = i
        for k in range(i, n):
            if abs(A[k][i]) > abs(A[max_row][i]):
                max_row = k

        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]

    return A, b


# ---------- MAIN APP ----------

class JacobiApp(App):

    def build(self):
        self.matrix_entries = []

        root = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Logo
        root.add_widget(Image(source="logo3.png", size_hint=(1, 0.25)))

        # Help button
        help_btn = Button(text="How to Use", size_hint=(1, None), height=40)
        help_btn.bind(on_press=self.show_help)
        root.add_widget(help_btn)

        # Inputs
        self.size_input = TextInput(
            hint_text="Matrix size n (n x n)",
            input_filter='int',
            size_hint=(1, None),
            height=40
        )
        root.add_widget(self.size_input)

        self.iter_input = TextInput(
            hint_text="Number of iterations (max 50)",
            input_filter='int',
            size_hint=(1, None),
            height=40
        )
        root.add_widget(self.iter_input)

        create_btn = Button(text="Create Matrix Input", size_hint=(1, None), height=40)
        create_btn.bind(on_press=self.create_matrix_inputs)
        root.add_widget(create_btn)

        # Matrix input area
        self.matrix_box = GridLayout(cols=1, spacing=5, size_hint_y=None)
        self.matrix_box.bind(minimum_height=self.matrix_box.setter('height'))

        scroll = ScrollView(size_hint=(1, 0.4))
        scroll.add_widget(self.matrix_box)
        root.add_widget(scroll)

        # Vector b
        self.vector_b = TextInput(
            hint_text="Vector b (space separated)",
            size_hint=(1, None),
            height=40
        )
        root.add_widget(self.vector_b)

        # Buttons
        btn_box = BoxLayout(size_hint=(1, None), height=40, spacing=10)
        solve_btn = Button(text="Solve")
        reset_btn = Button(text="Reset")

        solve_btn.bind(on_press=self.solve)
        reset_btn.bind(on_press=self.reset)

        btn_box.add_widget(solve_btn)
        btn_box.add_widget(reset_btn)
        root.add_widget(btn_box)

        # Output
        self.output = TextInput(readonly=True, size_hint=(1, 0.5))
        root.add_widget(self.output)

        return root

    # ---------- HELP POPUP ----------

    def show_help(self, instance):
        text = (
            "HOW TO USE:\n\n"
            "1. Enter matrix size n (n × n)\n"
            "2. Enter number of iterations (max 50)\n"
            "3. Click 'Create Matrix Input'\n"
            "4. Enter matrix A row by row\n"
            "5. Enter vector b\n"
            "6. Click Solve\n\n"
            "EXAMPLE:\n"
            "n = 3\n\n"
            "Matrix A:\n"
            "10 -1  2\n"
            "-1 11 -1\n"
            "2 -1 10\n\n"
            "Vector b:\n"
            "6 25 -11"
        )

        popup = Popup(
            title="How to Use – Jacobi Method",
            content=Label(text=text),
            size_hint=(0.9, 0.9)
        )
        popup.open()

    # ---------- MATRIX INPUT ----------

    def create_matrix_inputs(self, instance):
        self.matrix_box.clear_widgets()
        self.matrix_entries.clear()

        n = int(self.size_input.text)

        for i in range(n):
            entry = TextInput(
                hint_text=f"Row {i + 1}",
                size_hint_y=None,
                height=40
            )
            self.matrix_entries.append(entry)
            self.matrix_box.add_widget(entry)

    # ---------- SOLVER ----------

    def solve(self, instance):
        try:
            self.output.text = ""

            n = int(self.size_input.text)
            user_iter = int(self.iter_input.text)

            if user_iter < 1:
                raise ValueError("Iterations must be at least 1")

            iterations = min(user_iter, MAX_ITER)

            # Matrix A
            A = []
            for entry in self.matrix_entries:
                row = list(map(float, entry.text.split()))
                if len(row) != n:
                    raise ValueError("Each row must have exactly n values")
                A.append(row)

            # Vector b
            b = list(map(float, self.vector_b.text.split()))
            if len(b) != n:
                raise ValueError("Vector b must have exactly n values")

            A = np.array(A)
            b = np.array(b)

            self.output.text += "Matrix A:\n"
            for row in A:
                self.output.text += f"{row}\n"

            self.output.text += "\nVector b:\n"
            self.output.text += f"{b}\n\n"

            # Diagonal dominance
            if not is_diagonally_dominant(A):
                self.output.text += "Matrix not diagonally dominant.\nRearranging...\n\n"
                A, b = rearrange_to_diagonal_dominance(A, b)

                if not is_diagonally_dominant(A):
                    self.output.text += "Matrix cannot be rearranged.\n"
                    return

            # Jacobi iteration
            x = np.zeros(n)
            self.output.text += f"Jacobi Iterations (Total: {iterations}):\n"

            for itr in range(1, iterations + 1):
                x_new = np.zeros(n)
                for i in range(n):
                    s = sum(A[i][j] * x[j] for j in range(n) if j != i)
                    x_new[i] = (b[i] - s) / A[i][i]

                self.output.text += f"Iteration {itr}: {x_new}\n"
                x = x_new

        except Exception as e:
            self.output.text = str(e)

    # ---------- RESET ----------

    def reset(self, instance):
        self.size_input.text = ""
        self.iter_input.text = ""
        self.vector_b.text = ""
        self.output.text = ""
        self.matrix_box.clear_widgets()
        self.matrix_entries.clear()


if __name__ == "__main__":
    JacobiApp().run()
