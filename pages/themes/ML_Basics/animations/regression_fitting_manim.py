from manim import *
import numpy as np


class RegressionFitting(Scene):
    def construct(self):
        # ---------------------------------------------------------
        # 1. Setup the Coordinate System
        # ---------------------------------------------------------
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 12, 2],
            x_length=10,
            y_length=6,
            axis_config={"color": WHITE, "include_numbers": True},
            tips=False,
        )

        labels = axes.get_axis_labels(x_label="x", y_label="y")

        # ---------------------------------------------------------
        # 2. Generate Synthetic Data
        # ---------------------------------------------------------
        np.random.seed(123)
        x_data = np.linspace(1, 9, 10)
        # True relation: y = 1.0x + 1.5 + noise
        y_data = 1.0 * x_data + 1.5 + np.random.normal(0, 1.0, size=len(x_data))

        # Create Dots for the data
        dots = VGroup(
            *[Dot(axes.c2p(x, y), color=TEAL) for x, y in zip(x_data, y_data)]
        )

        # ---------------------------------------------------------
        # 3. Animation Logic (ValueTrackers)
        # ---------------------------------------------------------
        # We use ValueTrackers to control the Slope (m) and Intercept (b)
        # Start with a "bad" random guess
        m_tracker = ValueTracker(0.2)
        b_tracker = ValueTracker(8.0)

        # The Regression Line: Updates automatically when trackers change
        line = always_redraw(
            lambda: axes.plot(
                lambda x: m_tracker.get_value() * x + b_tracker.get_value(),
                color=RED,
                stroke_width=4,
                x_range=[0, 10],
            )
        )

        # The Residuals (Error Lines): Vertical lines from point to regression line
        residuals = always_redraw(
            lambda: self.get_residuals(
                axes, x_data, y_data, m_tracker.get_value(), b_tracker.get_value()
            )
        )

        # The Cost Text: Shows the Mean Squared Error
        cost_text = always_redraw(
            lambda: self.get_cost_text(
                x_data, y_data, m_tracker.get_value(), b_tracker.get_value()
            )
        )

        # ---------------------------------------------------------
        # 4. Construct the Scene
        # ---------------------------------------------------------
        self.add(axes, labels, dots)
        self.play(Create(line), FadeIn(residuals), Write(cost_text))
        self.wait(1)

        # Calculate the Optimal Fit (Ordinary Least Squares)
        A = np.vstack([x_data, np.ones(len(x_data))]).T
        m_opt, b_opt = np.linalg.lstsq(A, y_data, rcond=None)[0]

        # Animate from current (bad) values to optimal values
        self.play(
            m_tracker.animate.set_value(m_opt),
            b_tracker.animate.set_value(b_opt),
            run_time=5,
            rate_func=rate_functions.ease_in_out_cubic,  # Smooth start and end
        )

        # Highlight the result
        self.wait(1)
        final_box = SurroundingRectangle(cost_text, color=YELLOW, buff=0.2)
        self.play(Create(final_box))
        self.wait(2)

    def get_residuals(self, axes, x_data, y_data, m, b):
        """Helper to draw vertical lines for residuals"""
        group = VGroup()
        for x, y in zip(x_data, y_data):
            y_pred = m * x + b
            # Coordinate of the actual data point
            p_data = axes.c2p(x, y)
            # Coordinate of the point on the regression line
            p_pred = axes.c2p(x, y_pred)

            # Create a line connecting them
            line = Line(
                p_data, p_pred, color=YELLOW, stroke_opacity=0.6, stroke_width=2
            )
            group.add(line)
        return group

    def get_cost_text(self, x_data, y_data, m, b):
        """Helper to calculate and display MSE"""
        y_pred = m * x_data + b
        mse = np.mean((y_data - y_pred) ** 2)
        return Text(f"MSE Cost: {mse:.2f}", font_size=24, font="Monospace").to_corner(
            UR
        )
