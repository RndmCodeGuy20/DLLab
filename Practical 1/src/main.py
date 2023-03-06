from Gradient_Descent import GradientDescentFamily

if __name__ == '__main__':
    gd: GradientDescentFamily = GradientDescentFamily(0.0, 0.0, 0.05, 1.009)

    gd.Vanilla_Gradient_Descent()
    gd.Momentum_Gradient_Descent()
    gd.Show_Plot()
