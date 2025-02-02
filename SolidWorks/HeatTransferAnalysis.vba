Dim swApp As Object
Dim swDoc As Object
Dim swSim As Object
Dim swStudy As Object
Dim swHeatLoad As Object
Dim swConvect As Object

Sub HeatTransferAnalysis()
    ' Open SolidWorks
    Set swApp = Application.SldWorks
    Set swDoc = swApp.NewDocument("", 0, 0, 0)

    ' Activate Simulation
    Set swSim = swDoc.Extension.GetSimulation()
    Set swStudy = swSim.GetStudyManager().CreateNewStudy("Heat Transfer", 3) ' 3 = Thermal Analysis
    
    ' Apply Heat Load (500W)
    Set swHeatLoad = swStudy.CreateHeatPower()
    swHeatLoad.AddSelection "Face1"
    swHeatLoad.SetPowerValue 500
    
    ' Apply Convection (h = 25 W/m²K, T∞ = 300K)
    Set swConvect = swStudy.CreateConvection()
    swConvect.AddSelection "Face2"
    swConvect.SetConvectionParameters 25, 300
    
    ' Run Analysis
    swStudy.RunAnalysis
    
    ' Save Results
    swStudy.SaveResults "C:\Users\Public\Documents\ThermalResults.txt"
    
    MsgBox "Thermal Analysis Complete!"
End Sub
