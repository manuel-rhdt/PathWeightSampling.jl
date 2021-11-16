import PWS

system = PWS.cooperative_chemotaxis_system(dtimes=0:0.1:10)
algorithm = PWS.SMCEstimate(128)
result = PWS.mutual_information(system, algorithm, num_samples=1)
