import React, { useState, useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ScatterPlot, ResponsiveContainer, ScatterChart, Scatter, LineChart, Line, PieChart, Pie, Cell } from 'recharts';

const GalaxyAnalysisDashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  
  // Parse the CSV data
  const rawData = `name,alpha,chi2,reduced_chi2,r_squared,n_points,galaxy_size,ell_used
CamB,0.010004,85.298,10.662,0.302,9,1.790,0.448
D512-2,0.999262,5.541,1.847,0.406,4,3.830,0.958
D564-8,1.305709,55.387,11.077,0.136,6,3.070,0.767
D631-7,1.413847,735.115,49.008,-0.163,16,7.190,1.798
DDO064,0.754473,52.998,4.077,0.546,14,2.980,0.745
DDO168,1.052376,398.737,44.304,0.306,10,4.120,1.030
DDO170,1.586482,184.772,26.396,0.203,8,12.330,3.083
ESO079-G014,0.720301,416.439,29.746,0.539,15,16.670,4.168
ESO116-G012,1.161117,925.843,66.132,0.311,15,9.860,2.465
ESO444-G084,1.999996,510.651,85.108,-0.173,7,4.440,1.110
F561-1,0.010004,3.729,0.746,0.784,6,9.660,2.415
F563-1,1.802000,111.898,6.994,0.172,17,20.100,5.025
F563-V1,0.044975,0.870,0.174,0.906,6,7.870,1.968
F563-V2,0.999905,89.211,9.912,0.197,10,10.470,2.618
F565-V2,1.843249,30.764,5.127,0.179,7,8.800,2.200
F567-2,0.714308,2.759,0.690,0.602,5,9.590,2.397
F568-1,1.186911,112.354,10.214,0.337,12,13.230,3.308
F568-3,0.508221,193.791,11.399,0.232,18,17.980,4.495
F568-V1,1.806939,23.863,1.704,0.202,15,17.630,4.407
F571-V1,1.595174,21.732,3.622,0.167,7,13.590,3.397
F574-1,1.074494,234.168,18.013,0.447,14,12.600,3.150
F574-2,0.010004,3.360,0.840,0.438,5,10.830,2.708
F579-V1,1.023843,24.218,1.863,0.616,14,15.160,3.790
F583-1,1.074960,251.555,10.481,0.341,25,16.260,4.065
F583-4,0.908557,70.579,6.416,0.558,12,7.290,1.823
KK98-251,0.584259,130.076,9.291,0.656,15,3.130,0.782
NGC0024,1.579010,376.677,13.453,0.212,29,11.270,2.817
NGC0055,0.691477,479.349,23.967,0.438,21,13.500,3.375
NGC0100,0.860780,265.990,13.300,0.300,21,9.620,2.405
NGC0247,0.742236,865.030,34.601,0.124,26,14.540,3.635
NGC0289,1.376269,313.484,11.611,-14.930,28,71.120,17.780
NGC0300,1.476826,607.137,25.297,0.080,25,11.800,2.950
NGC0801,0.010006,686.667,57.222,-0.300,13,59.820,14.955
NGC1090,0.692911,894.149,38.876,0.397,24,30.090,7.522
NGC1705,1.999996,82.481,6.345,-3.265,14,6.000,1.500
NGC2366,0.902268,800.204,32.008,0.413,26,6.060,1.515
NGC2683,0.585564,111.796,11.180,-0.465,11,34.620,8.655
NGC2915,1.999996,306.162,10.557,-0.923,30,10.040,2.510
NGC2976,0.010006,28.407,1.093,0.978,27,2.270,0.568
NGC2998,0.845004,205.217,17.101,0.409,13,42.280,10.570
NGC3726,0.010004,257.353,23.396,-2.367,12,32.520,8.130
NGC3741,1.999996,528.205,26.410,-0.122,21,7.000,1.750
NGC3769,1.098637,145.280,13.207,-7.050,12,37.160,9.290
NGC3877,0.010005,278.388,23.199,0.462,13,11.350,2.837
NGC3893,0.010005,167.364,18.596,-5.105,10,19.050,4.763
NGC3917,0.613495,312.548,19.534,0.708,17,14.860,3.715
NGC3949,0.010006,196.690,32.782,-3.971,7,7.070,1.768
NGC3953,0.010005,69.150,9.879,-0.696,8,15.680,3.920
NGC3972,0.539769,88.135,9.793,0.738,10,8.720,2.180
NGC3992,0.921162,51.029,6.379,-0.431,9,46.020,11.505
NGC4010,0.427491,116.195,10.563,0.684,12,10.470,2.618
NGC4051,0.010004,104.766,17.461,-2.005,7,12.190,3.047
NGC4068,0.320631,22.337,4.467,0.762,6,2.330,0.583
NGC4085,0.010006,219.280,36.547,-0.066,7,6.200,1.550
NGC4088,0.010005,619.128,56.284,-2.826,12,21.480,5.370
NGC4100,0.732131,169.859,7.385,0.092,24,22.760,5.690
NGC4138,0.520634,23.446,3.908,-1.256,7,18.580,4.645
NGC4157,0.010005,360.680,22.542,-2.013,17,29.610,7.402
NGC4183,1.172726,82.156,3.734,0.417,23,21.020,5.255
NGC4214,1.647368,47.538,3.657,-1.370,14,5.630,1.407
NGC4389,0.010004,479.360,95.872,-3.376,6,5.320,1.330
NGC4559,0.721029,365.542,11.792,-0.150,32,20.970,5.242
NGC5005,0.010006,251.512,14.795,-12.174,18,11.470,2.868
NGC5907,0.859994,766.003,42.556,-4.737,19,50.330,12.582
NGC5985,1.023841,141.908,4.435,0.681,33,34.720,8.680
NGC6674,1.451546,397.954,28.425,-0.591,15,72.410,18.102
NGC6789,0.788478,26.331,8.777,0.214,4,0.710,0.177
NGC7793,0.010005,116.907,2.598,0.858,46,7.870,1.968
NGC7814,0.970613,537.441,31.614,-9.697,18,19.530,4.883
PGC51017,0.010006,19.474,3.895,-28.545,6,3.630,0.907
UGC00191,1.144021,976.504,122.063,0.286,9,9.980,2.495
UGC00634,1.949120,105.125,35.042,-0.271,4,18.010,4.503
UGC00731,1.415507,160.712,14.610,0.542,12,10.910,2.728
UGC00891,1.818783,318.104,79.526,-0.144,5,7.390,1.847
UGC01230,1.387337,49.027,4.903,0.286,11,36.540,9.135
UGC01281,0.895304,262.326,10.930,0.398,25,4.990,1.248
UGC02023,0.284570,4.828,1.207,0.684,5,3.780,0.945
UGC02259,1.607790,54.642,7.806,0.419,8,8.140,2.035
UGC02455,0.010005,755.875,107.982,-5.446,8,4.030,1.008
UGC02487,1.511326,177.802,11.113,-1.074,17,80.380,20.095
UGC02885,0.915065,282.101,15.672,-7.380,19,74.070,18.517
UGC04278,0.816402,318.063,13.253,0.457,25,6.690,1.673
UGC04305,0.010006,51.528,2.454,0.669,22,5.520,1.380
UGC04325,1.162745,100.708,14.387,0.581,8,5.590,1.397
UGC04483,0.888902,41.014,5.859,0.445,8,1.210,0.302
UGC04499,0.988449,104.945,13.118,0.386,9,8.180,2.045
UGC05005,1.052077,63.517,6.352,0.280,11,28.610,7.152
UGC05414,0.626774,69.352,13.870,0.641,6,4.110,1.028
UGC05716,1.747518,742.640,67.513,0.208,12,12.370,3.092
UGC05721,1.882867,272.291,12.377,0.012,23,6.740,1.685
UGC05750,0.560394,63.048,6.305,0.521,11,22.850,5.713
UGC05764,1.944840,359.265,39.918,0.168,10,3.620,0.905
UGC05829,0.891153,41.521,4.152,0.622,11,6.910,1.728
UGC05918,1.265654,19.028,2.718,0.509,8,4.460,1.115
UGC05986,1.192110,886.863,63.347,0.307,15,9.410,2.353
UGC05999,1.154789,52.715,13.179,0.317,5,16.220,4.055
UGC06399,1.142825,61.869,7.734,0.433,9,7.850,1.962
UGC06446,1.581656,69.630,4.352,0.322,17,10.220,2.555
UGC06614,0.582909,189.570,15.798,-9.414,13,64.590,16.148
UGC06628,0.010006,6.675,1.112,-1.428,7,7.690,1.923
UGC06667,1.841038,173.862,21.733,0.344,9,7.850,1.962
UGC06818,0.682797,81.190,11.599,0.195,8,6.980,1.745
UGC06917,0.907993,109.156,10.916,0.362,11,10.470,2.618
UGC06923,0.634512,24.975,4.995,0.275,6,5.160,1.290
UGC06930,1.066111,25.837,2.871,0.378,10,16.610,4.152
UGC06983,1.402860,78.348,4.897,0.004,17,15.680,3.920
UGC07089,0.504016,46.465,4.224,0.587,12,9.160,2.290
UGC07125,0.830955,98.684,8.224,0.474,13,18.680,4.670
UGC07151,0.627111,133.660,13.366,0.656,11,5.500,1.375
UGC07232,0.502222,19.689,6.563,0.491,4,0.820,0.205
UGC07261,1.022386,10.480,1.747,0.453,7,6.670,1.667
UGC07323,0.366859,45.973,5.108,0.835,10,5.820,1.455
UGC07399,1.810307,244.242,27.138,0.077,10,6.130,1.532
UGC07524,0.917959,463.721,15.457,0.557,31,10.690,2.672
UGC07559,0.571795,16.847,2.808,0.601,7,2.530,0.632
UGC07577,0.010004,2.793,0.349,0.847,9,1.690,0.422
UGC07603,1.500251,241.134,21.921,0.087,12,4.110,1.028
UGC07608,1.472225,36.773,5.253,0.316,8,4.780,1.195
UGC07690,0.849484,10.900,1.817,-0.252,7,4.130,1.032
UGC07866,0.700911,5.908,0.985,0.678,7,2.320,0.580
UGC08286,1.661221,436.729,27.296,0.267,17,8.040,2.010
UGC08490,1.999996,239.716,8.266,-0.275,30,10.150,2.538
UGC08550,1.532784,164.139,16.414,0.198,11,5.360,1.340
UGC08837,0.136396,55.829,7.976,0.700,8,4.200,1.050
UGC09037,0.010005,813.315,38.729,-0.613,22,27.960,6.990
UGC09992,0.622747,0.026,0.007,0.965,5,3.890,0.973
UGC10310,0.963151,18.161,3.027,0.631,7,7.740,1.935
UGC11557,0.010004,79.425,7.220,0.369,12,10.560,2.640
UGC11820,1.575867,670.319,74.480,-0.163,10,15.820,3.955
UGC11914,0.010006,962.437,15.038,0.343,65,9.830,2.458
UGC12506,1.198866,26.507,0.884,0.661,31,49.990,12.498
UGC12632,1.282778,75.043,5.360,0.452,15,10.660,2.665
UGC12732,1.354817,194.316,12.954,0.426,16,15.400,3.850
UGCA281,0.608921,35.701,5.950,0.682,7,1.080,0.270
UGCA442,1.999996,657.409,93.916,0.121,8,6.330,1.583
UGCA444,1.173140,126.828,3.624,0.268,36,2.620,0.655`;

  const galaxies = useMemo(() => {
    return rawData.split('\n').slice(1).map(line => {
      const [name, alpha, chi2, reduced_chi2, r_squared, n_points, galaxy_size, ell_used] = line.split(',');
      return {
        name,
        alpha: parseFloat(alpha),
        chi2: parseFloat(chi2),
        reduced_chi2: parseFloat(reduced_chi2),
        r_squared: parseFloat(r_squared),
        n_points: parseInt(n_points),
        galaxy_size: parseFloat(galaxy_size),
        ell_used: parseFloat(ell_used),
        fit_quality: parseFloat(r_squared) > 0.3 ? 'Good' : parseFloat(r_squared) > 0 ? 'Fair' : 'Poor',
        alpha_category: parseFloat(alpha) < 0.1 ? 'Low α' : parseFloat(alpha) > 1.5 ? 'High α' : 'Medium α'
      };
    });
  }, []);

  const stats = useMemo(() => {
    const goodFits = galaxies.filter(g => g.r_squared > 0.3).length;
    const fairFits = galaxies.filter(g => g.r_squared > 0 && g.r_squared <= 0.3).length;
    const poorFits = galaxies.filter(g => g.r_squared <= 0).length;
    
    const lowAlpha = galaxies.filter(g => g.alpha < 0.1).length;
    const midAlpha = galaxies.filter(g => g.alpha >= 0.1 && g.alpha <= 1.5).length;
    const highAlpha = galaxies.filter(g => g.alpha > 1.5).length;
    
    return {
      total: galaxies.length,
      goodFits,
      fairFits, 
      poorFits,
      lowAlpha,
      midAlpha,
      highAlpha,
      avgAlpha: galaxies.reduce((sum, g) => sum + g.alpha, 0) / galaxies.length,
      avgRSquared: galaxies.reduce((sum, g) => sum + g.r_squared, 0) / galaxies.length
    };
  }, [galaxies]);

  const fitQualityData = [
    { name: 'Good Fits (R² > 0.3)', value: stats.goodFits, fill: '#22c55e' },
    { name: 'Fair Fits (0 < R² ≤ 0.3)', value: stats.fairFits, fill: '#eab308' },
    { name: 'Poor Fits (R² ≤ 0)', value: stats.poorFits, fill: '#ef4444' }
  ];

  const alphaDistData = [
    { name: 'Low α (< 0.1)', value: stats.lowAlpha, fill: '#3b82f6' },
    { name: 'Medium α (0.1-1.5)', value: stats.midAlpha, fill: '#8b5cf6' },
    { name: 'High α (> 1.5)', value: stats.highAlpha, fill: '#f59e0b' }
  ];

  const renderOverview = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-blue-50 p-4 rounded-lg">
          <div className="text-2xl font-bold text-blue-600">{stats.total}</div>
          <div className="text-sm text-blue-800">Total Galaxies</div>
        </div>
        <div className="bg-green-50 p-4 rounded-lg">
          <div className="text-2xl font-bold text-green-600">{stats.goodFits}</div>
          <div className="text-sm text-green-800">Good Fits</div>
        </div>
        <div className="bg-purple-50 p-4 rounded-lg">
          <div className="text-2xl font-bold text-purple-600">{stats.avgAlpha.toFixed(3)}</div>
          <div className="text-sm text-purple-800">Avg α</div>
        </div>
        <div className="bg-orange-50 p-4 rounded-lg">
          <div className="text-2xl font-bold text-orange-600">{(stats.avgRSquared * 100).toFixed(1)}%</div>
          <div className="text-sm text-orange-800">Avg R²</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Fit Quality Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={fitQualityData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({name, percent}) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {fitQualityData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Alpha Parameter Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={alphaDistData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({name, percent}) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {alphaDistData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );

  const renderScatterPlots = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Alpha vs Galaxy Size</h3>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart data={galaxies}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="galaxy_size" name="Galaxy Size (kpc)" />
              <YAxis dataKey="alpha" name="Alpha" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Scatter name="Galaxies" fill="#8884d8" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">R² vs Alpha</h3>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart data={galaxies}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="alpha" name="Alpha" />
              <YAxis dataKey="r_squared" name="R²" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Scatter name="Fit Quality" fill="#82ca9d" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-white p-4 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Galaxy Size Distribution</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={galaxies.slice(0, 50)}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="galaxy_size" fill="#8884d8" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const renderPatternAnalysis = () => {
    const extremeAlphaGalaxies = galaxies.filter(g => g.alpha < 0.05 || g.alpha > 1.9);
    const excellentFits = galaxies.filter(g => g.r_squared > 0.8);
    const poorFits = galaxies.filter(g => g.r_squared < 0);
    
    return (
      <div className="space-y-6">
        <div className="bg-red-50 p-4 rounded-lg border border-red-200">
          <h3 className="text-lg font-semibold text-red-800 mb-4">🚨 Extreme Alpha Values (Boundary Hits)</h3>
          <div className="text-sm text-red-700 mb-2">
            These {extremeAlphaGalaxies.length} galaxies hit parameter boundaries (α ≈ 0.01 or α ≈ 2.0):
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {extremeAlphaGalaxies.slice(0, 20).map(g => (
              <div key={g.name} className="bg-white p-2 rounded">
                {g.name}: α={g.alpha.toFixed(3)}, R²={g.r_squared.toFixed(3)}
              </div>
            ))}
          </div>
        </div>

        <div className="bg-green-50 p-4 rounded-lg border border-green-200">
          <h3 className="text-lg font-semibold text-green-800 mb-4">✨ Excellent Fits (R² > 0.8)</h3>
          <div className="text-sm text-green-700 mb-2">
            These {excellentFits.length} galaxies show exceptional agreement:
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {excellentFits.map(g => (
              <div key={g.name} className="bg-white p-2 rounded">
                {g.name}: α={g.alpha.toFixed(3)}, R²={g.r_squared.toFixed(3)}
              </div>
            ))}
          </div>
        </div>

        <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
          <h3 className="text-lg font-semibold text-yellow-800 mb-4">⚠️ Problematic Fits (R² < 0)</h3>
          <div className="text-sm text-yellow-700 mb-2">
            These {poorFits.length} galaxies show poor fits - potential morphology issues:
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {poorFits.slice(0, 20).map(g => (
              <div key={g.name} className="bg-white p-2 rounded">
                {g.name}: α={g.alpha.toFixed(3)}, R²={g.r_squared.toFixed(3)}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gray-50">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">🌌 Galaxy Analysis Results</h1>
        <p className="text-gray-600">Analysis of 136 successful fits from your geodesic theory</p>
      </div>

      <div className="mb-6">
        <div className="flex space-x-4 border-b">
          {[
            { id: 'overview', label: '📊 Overview' },
            { id: 'scatter', label: '📈 Scatter Plots' },
            { id: 'patterns', label: '🔍 Pattern Analysis' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2 font-medium transition-colors ${
                activeTab === tab.id
                  ? 'text-blue-600 border-b-2 border-blue-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-sm">
        {activeTab === 'overview' && renderOverview()}
        {activeTab === 'scatter' && renderScatterPlots()}
        {activeTab === 'patterns' && renderPatternAnalysis()}