# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2012 Dominik Kriegner <dominik.kriegner@gmail.com>

import xrayutilities as xu
import numpy
import matplotlib.pyplot as plt
import matplotlib as mpl
import os.path
from mpl_toolkits.basemap import *

sample = "test"
datadir = "data"
basename = sample

eps = 0.01
# plot settings
mpl.rcParams['font.size'] = 18.0
mpl.rcParams['axes.labelsize'] = 'large'
mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.handletextpad'] = 0.5
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['figure.subplot.bottom'] = 0.13
mpl.rcParams['figure.subplot.top'] = 0.93
mpl.rcParams['figure.subplot.left'] = 0.14
mpl.rcParams['figure.subplot.right'] = 0.915
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['axes.grid'] = False

#substrate
Ge = xu.materials.Ge

tup113 = (Ge.Q(1,1,3),Ge.Q(1,3,1),\
          Ge.Q(3,1,1),Ge.Q(-1,1,3),\
          Ge.Q(-1,3,1),Ge.Q(3,1,-1),\
          Ge.Q(1,-1,3),Ge.Q(1,3,-1),\
          Ge.Q(3,-1,1),Ge.Q(1,1,-3),\
          Ge.Q(1,-3,1),Ge.Q(-3,1,1),\
          Ge.Q(-1,-1,-3),Ge.Q(-1,-3,-1),\
          Ge.Q(-3,-1,-1),Ge.Q(1,-1,-3),\
          Ge.Q(1,-3,-1),Ge.Q(-3,-1,1),\
          Ge.Q(-1,1,-3),Ge.Q(-1,-3,1),\
          Ge.Q(-3,1,-1),Ge.Q(-1,-1,3),\
          Ge.Q(-1,3,-1),Ge.Q(3,-1,-1))

label113 = (r'$(113)$',r'$(131)$',r'$(311)$',
          r'$(\bar 113)$',r'$(\bar 131)$',r'$(31\bar 1)$',
          r'$(1\bar 13)$',r'$(13\bar 1)$',r'$(3\bar 11)$',
          r'$(11\bar 3)$',r'$(1\bar 31)$',r'$(\bar 311)$',
          r'$(\bar 1\bar 1\bar 3)$',r'$(\bar 1\bar 3\bar 1)$',r'$(\bar 3\bar 1\bar 1)$',
          r'$(1\bar 1\bar 3)$',r'$(1\bar 3\bar 1)$',r'$(\bar 3\bar 11)$',
          r'$(\bar 1 1\bar 3)$',r'$(\bar 1\bar 3 1)$',r'$(\bar 3 1\bar 1)$',
          r'$(\bar 1\bar 1 3)$',r'$(\bar 1 3\bar 1)$',r'$( 3\bar 1\bar 1)$')

df = xu.io.XRDMLFile(os.path.join(datadir,"polefig_Ge113.xrdml"))
s = df.scan

chi = 90-s['Psi']
phi = s['Phi']
INT = s['detector']

#create 2D arrays for the angles
CHI = chi[:,numpy.newaxis]*numpy.ones(INT.shape)
PHI = phi

INT = xu.maplog(INT,6,0)

fig = plt.figure(); plt.clf()
#plt.title("(113) pole figure")
m = Basemap(boundinglat=-1.,lon_0=180.0,resolution=None,projection='npstere')
X,Y = m(PHI,CHI)
ax = plt.subplot(111)
ax.set_frame_on(False)
CS = m.contourf(X,Y,INT,50)
m.drawparallels(numpy.arange(0,91,10),labels=[1,1,1,1],color='gray',dashes=[2,2])
m.drawmeridians(numpy.arange(0,360,60),labels=[1,1,1,1],color='gray',labelstyle='+/-',dashes=[2,2])

dphi = 62
dchi = -32
inpdir = xu.math.rotarb(Ge.Q(1,-1,0),Ge.Q(0,0,1),dphi)
ndir = xu.math.rotarb(Ge.Q(0,0,1), inpdir, dchi)

# plot normaldir
# calculate spherical coordinate angles
[chi,phi]=xu.analysis.getangles(Ge.Q(0,0,1),ndir,inpdir)
chi = 90-chi
phi = phi-dphi
# if direction is visible plot in polefig
if (chi >= -eps):
    x,y = m(phi,chi)
    m.plot(numpy.array([x]),numpy.array([y]),ls='None',marker='s',color='k',ms=12.)

# plot Ge {113} Bragg peaks
for i in range(len(tup113)):
    dir = tup113[i]
    # calculate spherical coordinate angles
    dir = xu.math.rotarb(dir,Ge.Q(0,0,1),27)
    [chi,phi]=xu.analysis.getangles(dir,ndir,inpdir)
    chi = 90-chi
    phi = phi-dphi
    # if direction is visible plot in polefig
    if (chi >= -eps):
        x,y = m(phi,chi)
        m.plot(numpy.array([x]),numpy.array([y]),ls='None',marker='o',color='k',ms=8.,mfc='None',mew=2.)
        plt.text(x+400000,y,label113[i],ha='left',va='bottom')

