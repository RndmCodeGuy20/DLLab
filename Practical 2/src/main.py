from Gradient_Descent_AdaGrad_RMSProp import GradientDescentFamilyUpdated

if __name__ == '__main__':
    gd: GradientDescentFamilyUpdated = GradientDescentFamilyUpdated(0.0, 0.0, 0.01, 0.3, 0.4)

    gd.Plot_MeshGrid()
    # gd.AdaGrad(13, 8)
    gd.RMSProp(-0.00000013, -0.00000025)
    gd.Show_Plot()
