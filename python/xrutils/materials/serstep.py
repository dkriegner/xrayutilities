#Python module to connect to Sergey Stepano's x-ray server
#'http://sergey.gmca.aps.anl.gov/
#In particular to the Xhi_0 interface. It should not only 
#fetch some basic data but also parse the output of the website

import urllib2 as url
import ClientForm as cf

chi_0_url = "http://sergey.gmca.aps.anl.gov/cgi/WWW_form.exe?template=x0h_form.htm"

class SerStepRequest(object):
    def __init__(self):
        source = url.urlopen(chi_0_url)
        forms = cf.ParseResponse(source,backwards_compat=False)
        form = forms[0]
        source.close()
            
    def __call__(self,**keyargs):
        if keyargs.has_key("en"):
            form["xway"]=["2"]
            form["wave"]="%f" %(keyargs["en"])
            
        if keyargs.has_key("wl"):
            form["xway"]=["1"]
            form["wave"]="%f" %(keyargs["wl"])
            
        if keyargs.has_key("cl"):
            form["xway"]=["3"]
            try:
                form["line"]=keyargs["cl"]
            except:
                print "unknown characteristic line!!!"
                return None
                
        if keyargs.has_key("cryst"):
            #select a crystaline material
            form["coway"] = ["0"]
            try:
                form["code"]=keyargs["cryst"]
            except:
                print "unknown crystal code!"
                return None
        
        if keyargs.has_key("amor"):
            #select an amorphous material
            form["coway"] =["1"]
            try:
                form["amor"]=keyargs["amor"]
            except:
                print "unknown amorphous material!"
                return None
                
        if keyargs.has_key("cf") and keyargs.has_key("rho"):
            form["coway"] = ["2"]
            form["chem"] = keyargs["cf"]
            form["rho"] = keyargs["rho"]
            
        if keyargs.has_key("hkl"):
            form["i1"] = "%f" %(keyargs["hkl"][0])
            form["i2"] = "%f" %(keyargs["hkl"][1])
            form["i3"] = "%f" %(keyargs["hkl"][2])
            
        #set general option that the result will be returned in text form
        form["modeout"] = "2"
        
        #send the request
        res = url.urlopen(form.click()).read()          
        print res
    
    def __str__(self):
        pass
    
