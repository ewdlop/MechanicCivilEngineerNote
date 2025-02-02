using System;
using System.Runtime.InteropServices;
using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;
using SolidWorks.Interop.simulation;

class Program
{
    static void Main()
    {
        SldWorks swApp = (SldWorks)Marshal.GetActiveObject("SldWorks.Application");
        ModelDoc2 swDoc = (ModelDoc2)swApp.NewDocument(@"C:\ProgramData\SolidWorks\SOLIDWORKS 2022\templates\Part.prtdot", 0, 0, 0);
        SketchManager swSketchMgr = swDoc.SketchManager;
        FeatureManager swFeatureMgr = swDoc.FeatureManager;

        // Define gear parameters
        double pitchDiameter = 0.1;  // 100mm
        double thickness = 0.02;     // 20mm
        int numTeeth = 20;

        // Create Base Circle for Gear
        swSketchMgr.InsertSketch(true);
        swSketchMgr.CreateCircle(0, 0, 0, pitchDiameter / 2, 0, 0);
        swSketchMgr.InsertSketch(false);
        swFeatureMgr.FeatureExtrusion2(true, false, false, 0, 0, thickness, 0, false, false, false, false, 0, 0, false, false, false, false, true, true, true, 0, 0, false);

        // Apply FEA
        Simulation swSim = (Simulation)swDoc.GetSimulation();
        StudyManager swStudyMgr = swSim.GetStudyManager();
        StaticStudy swStudy = (StaticStudy)swStudyMgr.CreateNewStudy("Gear Load Analysis", (int)swStudyType_e.swAnalysisStudyStatic);
        
        // Apply Material (Steel)
        Material swMaterial = swStudy.GetPart().GetMaterial();
        swMaterial.SetMaterial("SOLIDWORKS Materials.sldmat", "Steel");

        // Apply Force (Torque)
        Load swTorque = swStudy.CreateTorque();
        swTorque.AddSelection("Face1"); // Apply torque to face of gear
        swTorque.SetTorqueValue(500);   // 500 Nm torque

        // Run Simulation
        swStudy.RunAnalysis();

        // Save Results
        swStudy.SaveResults(@"C:\Users\Public\Documents\GearStressResults.txt");

        Console.WriteLine("Spur gear FEA complete! Results saved.");
    }
}
