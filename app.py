
from flask import Flask, render_template, request, jsonify
import math
import re

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

app = Flask(__name__)

# Symbols used by SymPy
x, y = sp.symbols("x y")

# Supported mathematical functions
ALLOWED_FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "pi": sp.pi,
    "E": sp.E,
}

# Numerical methods available on the website
METHODS = {
    "euler": "Euler's Method",
    "heun": "Improved Euler (Heun)",
    "midpoint": "Midpoint Method (RK2)",
    "rk3": "Runge-Kutta 3rd Order (RK3)",
    "rk4": "Runge-Kutta 4th Order (RK4)",
    "ralston": "Ralston's Method (RK2)",
    "taylor": "Taylor Series Method (order 4)",
    "ab4": "Adams-Bashforth (4-step)",
    "am4": "Adams-Moulton (4-step predictor-corrector)",
    "backward": "Backward Euler",
    "trapezoidal": "Trapezoidal (implicit)",
    "exact": "Exact Analytical Solution",
    "scipy": "SciPy solve_ivp (RK45)",
}


# --------------------------------------------------
# 1. Read and validate the differential equation
# --------------------------------------------------

def make_equation(text):
    text = str(text).strip()

    allowed_chars = re.compile(
        r"^[0-9xyXYeE+\-*/().,\s^a-zA-Z]+$"
    )

    if not text or len(text) > 160:
        raise ValueError("Enter a valid differential equation.")

    if not allowed_chars.fullmatch(text) or "__" in text:
        raise ValueError("Invalid mathematical expression.")

    # Allow users to type x^2 as well as x**2
    text = text.replace("^", "**")

    try:
        expr = sp.sympify(
            text,
            locals={
                **ALLOWED_FUNCTIONS,
                "x": x,
                "y": y,
            },
        )
    except Exception as exc:
        raise ValueError(
            "Cannot understand the equation. Check your input."
        ) from exc

    if not expr.free_symbols.issubset({x, y}):
        raise ValueError("Only x and y are allowed as variables.")

    allowed_functions = {
        sp.sin,
        sp.cos,
        sp.tan,
        sp.exp,
        sp.log,
        sp.sqrt,
        sp.Abs,
    }

    for node in sp.preorder_traversal(expr):
        if node.is_Function and node.func not in allowed_functions:
            raise ValueError(
                "Supported functions: sin, cos, tan, exp, log, sqrt, abs."
            )

    function = sp.lambdify(
        (x, y),
        expr,
        modules=["numpy"],
    )

    def f(x_value, y_value):
        try:
            result = float(function(float(x_value), float(y_value)))
        except Exception as exc:
            raise ValueError(
                "The equation is undefined at a calculation point."
            ) from exc

        if not math.isfinite(result):
            raise ValueError(
                "The equation produced an invalid numerical value."
            )

        return result

    return expr, f


# --------------------------------------------------
# 2. Exact analytical solution
# --------------------------------------------------

def exact_solution(expr, x0, y0):
    Y = sp.Function("Y")
    dependent_y = Y(x)

    equation_rhs = expr.subs(y, dependent_y)

    try:
        solution = sp.dsolve(
            sp.Eq(
                sp.diff(dependent_y, x),
                equation_rhs,
            ),
            ics={
                dependent_y.subs(x, x0): y0
            },
        )

        result = solution.rhs

        if result.has(sp.Integral, sp.Derivative):
            return None

        return result

    except Exception:
        return None


def exact_values(solution, xs):
    if solution is None:
        return None

    try:
        function = sp.lambdify(
            x,
            solution,
            modules=["numpy"],
        )

        values = np.asarray(
            function(np.asarray(xs, dtype=float))
        )

        if values.ndim == 0:
            values = np.full(len(xs), float(values))

        values = values.astype(float).reshape(-1)

        if len(values) != len(xs):
            return None

        if not np.all(np.isfinite(values)):
            return None

        return values

    except Exception:
        return None


# --------------------------------------------------
# 3. Create x values from the website inputs
# --------------------------------------------------

def make_grid(x0, end, h):
    count = int(math.ceil((end - x0) / h))

    if count > 10000:
        raise ValueError(
            "Too many steps. Increase the step size."
        )

    xs = [x0 + i * h for i in range(count)]
    xs = [value for value in xs if value < end]

    # Always include the requested final x value.
    xs.append(end)

    return np.asarray(xs, dtype=float)


# --------------------------------------------------
# 4. One step of Euler / Runge-Kutta methods
# --------------------------------------------------

def rk_step(f, a, z, h, method="rk4"):
    """
    a = current x
    z = current y
    h = step size

    IMPORTANT: argument order is f, x, y, h, method.
    """

    a = float(a)
    z = float(z)
    h = float(h)

    k1 = f(a, z)

    if method == "euler":
        return z + h * k1

    if method == "heun":
        k2 = f(a + h, z + h * k1)
        return z + h * (k1 + k2) / 2

    if method == "midpoint":
        k2 = f(
            a + h / 2,
            z + h * k1 / 2,
        )
        return z + h * k2

    if method == "rk3":
        k2 = f(
            a + h / 2,
            z + h * k1 / 2,
        )

        k3 = f(
            a + h,
            z - h * k1 + 2 * h * k2,
        )

        return z + h * (k1 + 4 * k2 + k3) / 6

    if method == "ralston":
        k2 = f(
            a + 2 * h / 3,
            z + 2 * h * k1 / 3,
        )

        return z + h * (k1 / 4 + 3 * k2 / 4)

    # Classical fourth-order Runge-Kutta
    k2 = f(a + h / 2, z + h * k1 / 2)
    k3 = f(a + h / 2, z + h * k2 / 2)
    k4 = f(a + h, z + h * k3)

    return z + h * (
        k1 + 2 * k2 + 2 * k3 + k4
    ) / 6


# --------------------------------------------------
# 5. Fixed-step numerical methods
# --------------------------------------------------

def fixed_step(f, xs, y0, method):
    ys = [float(y0)]

    for i in range(len(xs) - 1):
        step = float(xs[i + 1] - xs[i])

        # Correct argument order
        next_y = rk_step(
            f,
            float(xs[i]),
            float(ys[-1]),
            step,
            method,
        )

        if not math.isfinite(float(next_y)):
            raise ValueError(
                "The numerical method produced an invalid value."
            )

        ys.append(float(next_y))

    return np.asarray(ys, dtype=float)


# --------------------------------------------------
# 6. Fourth-order Taylor method
# --------------------------------------------------

def taylor4(expr, xs, y0):
    def total_derivative(q):
        return sp.diff(q, x) + expr * sp.diff(q, y)

    derivatives = [expr]

    for _ in range(3):
        derivatives.append(
            sp.simplify(total_derivative(derivatives[-1]))
        )

    functions = [
        sp.lambdify((x, y), derivative, modules=["numpy"])
        for derivative in derivatives
    ]

    ys = [float(y0)]

    for i in range(len(xs) - 1):
        current_x = float(xs[i])
        current_y = float(ys[-1])
        step = float(xs[i + 1] - xs[i])

        result = current_y

        for order, function in enumerate(functions, start=1):
            derivative_value = float(
                function(current_x, current_y)
            )

            if not math.isfinite(derivative_value):
                raise ValueError(
                    "Taylor derivative is not finite."
                )

            result += (
                step**order
                / math.factorial(order)
                * derivative_value
            )

        if not math.isfinite(result):
            raise ValueError("Taylor method produced invalid values.")

        ys.append(result)

    return np.asarray(ys, dtype=float)


# --------------------------------------------------
# 7. Adams-Bashforth and Adams-Moulton methods
# --------------------------------------------------

def multistep(f, xs, y0, kind):
    if len(xs) == 1:
        return np.asarray([float(y0)])

    # RK4 startup values
    startup_count = min(4, len(xs))

    ys = list(
        fixed_step(
            f,
            xs[:startup_count],
            y0,
            "rk4",
        )
    )

    slopes = [
        f(float(xs[i]), float(ys[i]))
        for i in range(len(ys))
    ]

    for i in range(len(ys) - 1, len(xs) - 1):
        step = float(xs[i + 1] - xs[i])

        if i < 3:
            next_y = rk_step(
                f,
                xs[i],
                ys[i],
                step,
                "rk4",
            )

        elif kind == "ab4":
            next_y = ys[i] + step * (
                55 * slopes[i]
                - 59 * slopes[i - 1]
                + 37 * slopes[i - 2]
                - 9 * slopes[i - 3]
            ) / 24

        else:
            # Adams-Bashforth predictor
            predictor = ys[i] + step * (
                55 * slopes[i]
                - 59 * slopes[i - 1]
                + 37 * slopes[i - 2]
                - 9 * slopes[i - 3]
            ) / 24

            next_y = predictor

            # Adams-Moulton corrector
            for _ in range(50):
                corrected = ys[i] + step * (
                    9 * f(xs[i + 1], next_y)
                    + 19 * slopes[i]
                    - 5 * slopes[i - 1]
                    + slopes[i - 2]
                ) / 24

                if abs(corrected - next_y) < 1e-11 * (
                    1 + abs(corrected)
                ):
                    next_y = corrected
                    break

                next_y = corrected

        if not math.isfinite(float(next_y)):
            raise ValueError(
                "Multistep method produced invalid values."
            )

        ys.append(float(next_y))
        slopes.append(f(float(xs[i + 1]), float(next_y)))

    return np.asarray(ys, dtype=float)


# --------------------------------------------------
# 8. Implicit methods
# --------------------------------------------------

def implicit_method(f, xs, y0, trapezoidal=False):
    ys = [float(y0)]

    for i in range(len(xs) - 1):
        a = float(xs[i])
        b = float(xs[i + 1])
        z = float(ys[-1])
        step = b - a

        fa = f(a, z)
        guess = z + step * fa

        for _ in range(100):
            if trapezoidal:
                next_guess = z + step * (
                    fa + f(b, guess)
                ) / 2
            else:
                next_guess = z + step * f(b, guess)

            if not math.isfinite(float(next_guess)):
                break

            if abs(next_guess - guess) < 1e-11 * (
                1 + abs(next_guess)
            ):
                guess = next_guess
                break

            guess = next_guess
        else:
            raise ValueError(
                "Implicit method did not converge. "
                "Try a smaller step size."
            )

        if not math.isfinite(float(guess)):
            raise ValueError(
                "Implicit method produced invalid values."
            )

        ys.append(float(guess))

    return np.asarray(ys, dtype=float)


# --------------------------------------------------
# 9. Select and execute a numerical method
# --------------------------------------------------

def calculate(method, expr, f, xs, y0):
    if method in {
        "euler", "heun", "midpoint",
        "rk3", "rk4", "ralston",
    }:
        return fixed_step(f, xs, y0, method)

    if method == "taylor":
        return taylor4(expr, xs, y0)

    if method in {"ab4", "am4"}:
        # Multistep formulas require equally spaced x values.
        steps = np.diff(xs)

        if len(steps) > 1 and not np.allclose(
            steps, steps[0], rtol=1e-8, atol=1e-12
        ):
            raise ValueError(
                "Adams methods require the interval to be an "
                "integer multiple of h. Adjust the step size "
                "or end x."
            )

        return multistep(f, xs, y0, method)

    if method == "backward":
        return implicit_method(f, xs, y0, False)

    if method == "trapezoidal":
        return implicit_method(f, xs, y0, True)

    if method == "scipy":
        solution = solve_ivp(
            lambda a, z: [f(a, z[0])],
            (float(xs[0]), float(xs[-1])),
            [float(y0)],
            method="RK45",
            t_eval=xs,
            rtol=1e-8,
            atol=1e-10,
        )

        if not solution.success:
            raise ValueError(
                "SciPy solver failed: " + solution.message
            )

        return solution.y[0]

    raise ValueError("Unknown numerical method.")


# --------------------------------------------------
# 10. Website home page
# --------------------------------------------------

@app.route("/")
def home():
    return render_template(
        "index.html",
        methods=METHODS,
    )


# --------------------------------------------------
# 11. Receive current inputs from the website
# --------------------------------------------------

@app.route("/solve", methods=["POST"])
def solve():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            raise ValueError("No form data received.")

        # These values come from the browser form.
        equation = str(data.get("equation", "")).strip()
        method = str(data.get("method", "")).strip().lower()

        if method != "all" and method not in METHODS:
            raise ValueError("Select a valid numerical method.")

        # Convert browser input strings into numbers.
        x0 = float(data["x0"])
        y0 = float(data["y0"])
        h = float(data["h"])
        end = float(data["end"])

        if not all(
            math.isfinite(value)
            for value in (x0, y0, h, end)
        ):
            raise ValueError("All inputs must be finite numbers.")

        if h <= 0:
            raise ValueError("Step size must be greater than zero.")

        if end <= x0:
            raise ValueError(
                "End x must be greater than initial x."
            )

        if (end - x0) / h > 10000:
            raise ValueError(
                "Too many steps. Increase the step size."
            )

        # Parse the equation entered on the website.
        expr, f = make_equation(equation)

        # Create the calculation points.
        xs = make_grid(x0, end, h)

        # Try to find the exact analytical solution.
        solution = exact_solution(expr, x0, y0)
        exact = exact_values(solution, xs)

        if method == "all":
            selected_methods = list(METHODS.keys())
        else:
            selected_methods = [method]

        results = {}
        errors = {}

        for name in selected_methods:
            if name == "exact":
                if exact is None:
                    results[name] = {
                        "label": METHODS[name],
                        "rows": [],
                        "available": False,
                        "message": (
                            "No exact solution was found automatically."
                        ),
                    }
                    continue

                values = exact.copy()

            else:
                values = np.asarray(
                    calculate(name, expr, f, xs, y0),
                    dtype=float,
                )

            if not np.all(np.isfinite(values)):
                raise ValueError(
                    METHODS[name] + " produced invalid values."
                )

            rows = []

            for i, (xx, yy) in enumerate(zip(xs, values)):
                row = {
                    "step": i,
                    "x": float(xx),
                    "y": float(yy),
                }

                if exact is not None:
                    row["exact"] = float(exact[i])
                    row["abs_error"] = abs(
                        float(yy) - float(exact[i])
                    )

                rows.append(row)

            results[name] = {
                "label": METHODS[name],
                "rows": rows,
                "available": True,
            }

            if exact is not None and name != "exact":
                errors[name] = {
                    "max_abs_error": float(
                        np.max(np.abs(values - exact))
                    ),
                    "final_abs_error": float(
                        abs(values[-1] - exact[-1])
                    ),
                }

        return jsonify({
            "equation": equation,
            "method": method,
            "exact_available": exact is not None,
            "exact_expression": (
                str(solution) if exact is not None else None
            ),
            "results": results,
            "errors": errors,
        })

    except (
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        ZeroDivisionError,
    ) as exc:
        return jsonify({"error": str(exc)}), 400

    except Exception:
        app.logger.exception("Solver error")

        return jsonify({
            "error": (
                "Calculation failed. Check the equation "
                "and try again."
            )
        }), 400


# --------------------------------------------------
# 12. Start Flask
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
