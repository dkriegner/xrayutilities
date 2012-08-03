import xrutils as xu
import numpy

Si = xu.materials.Si  # load material from materials submodule

# initialize experimental class with directions from experiment
exp = xu.HXRD(Si.Q(1,1,-2),Si.Q(1,1,1))

# calculate angles and print them to the screen
angs = exp.Q2Ang(Si.Q(1,1,1))
print(("|F000|: %8.3f" %(numpy.abs(Si.StructureFactor(Si.Q(0,0,0),exp.energy)))))

print("Si (111)")
print(("phi:%8.4f" %angs[2]))
print(("om: %8.4f" %angs[0]))
print(("tt: %8.4f" %angs[3]))
print(("|F|: %8.3f" %(numpy.abs(Si.StructureFactor(Si.Q(1,1,1),exp.energy)))))


angs = exp.Q2Ang(Si.Q(2,2,4))
print("Si (224)")
print("phi:%8.4f" %angs[2])
print("om: %8.4f" %angs[0])
print("tt: %8.4f" %angs[3])
print("|F|: %8.3f" %(numpy.abs(Si.StructureFactor(Si.Q(2,2,4),exp.energy))))


angs = exp.Q2Ang(Si.Q(3,3,1))
print("Si (331)")
print("phi:%8.4f" %angs[2])
print("om: %8.4f" %angs[0])
print("tt: %8.4f" %angs[3])
print("|F|: %8.3f" %(numpy.abs(Si.StructureFactor(Si.Q(3,3,1),exp.energy))))
