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
        ModelDoc2 swDoc = (ModelDoc2)swApp.ActiveDoc;

        // Activate Simulation
        Simulation swSim = (Simulation)swDoc.GetSimulation();
        StudyManager swStudyMgr = swSim.GetStudyManager();
        StaticStudy swStudy = (StaticStudy)swStudyMgr.CreateNewStudy("Bolt Joint Analysis", (int)swStudyType_e.swAnalysisStudyStatic);

        // Apply Bolt Pretension
        Load swBoltLoad = swStudy.CreateBoltPretension();
        swBoltLoad.AddSelection("Bolt1"); // Apply preload to first bolt
        swBoltLoad.SetPretensionValue(10000); // 10,000 N preload

        // Run Analysis
        swStudy.RunAnalysis();

        // Save Results
        swStudy.SaveResults(@"C:\Users\Public\Documents\BoltStressResults.txt");

        Console.WriteLine("Bolted joint stress analysis complete! Results saved.");
    }
}
