from Gradient_Descent_AdaGrad_RMSProp import GradientDescentFamilyUpdated

if __name__ == '__main__':
    gd: GradientDescentFamilyUpdated = GradientDescentFamilyUpdated(0.0, 0.0, 0.01, 0.3, 0.4)

    gd.Plot_MeshGrid(0.0, 0.0)
    # gd.AdaGrad(10, 10)
    gd.RMSProp(10, 10)
    gd.Show_Plot()
