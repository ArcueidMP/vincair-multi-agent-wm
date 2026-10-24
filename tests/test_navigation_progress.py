import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/e4_navigation'))
import diagnose_navigation_progress as d

class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.s=np.array([[-1,0,0,0],[1,0,0,0]],np.float32);self.g=-self.s[:,:2]
        self.end=np.array([self.s[:,:2],self.s[:,:2]*.5,self.s[:,:2]*.25]);self.raw=np.array([[-4,-1,1.2],[-4,0,1.2],[-4,1,1.2]],np.float32)
        self.c=dict(cost_mean=0,cost_std=1,policy=dict(threshold=.5,margin=0))
    def run_case(self,raw=None,minimum=None):return d.analyze(self.s,self.g,self.raw if raw is None else raw,np.array([3.,2.,1.]),np.full(3,.4) if minimum is None else minimum,np.zeros(3),self.end,self.c)
    def test_cost_ranking_failure_with_progress_coverage(self):
        r=self.run_case();self.assertTrue(r['safe_progress_available']);self.assertTrue(r['cost_miss_within_filter']);self.assertFalse(r['filter_excludes_all_progress'])
        self.assertEqual(r['methods']['learned']['index'],0);self.assertEqual(r['methods']['true_cost_predicted_filter']['index'],2)
    def test_filter_excludes_progress(self):
        raw=self.raw.copy();raw[1:,0]=4;r=self.run_case(raw)
        self.assertTrue(r['filter_excludes_all_progress']);self.assertFalse(r['cost_miss_within_filter'])
    def test_no_safe_progress_not_filter_failure(self):
        r=self.run_case(minimum=np.full(3,.2));self.assertFalse(r['safe_progress_available']);self.assertFalse(r['filter_excludes_all_progress'])
    def test_empty_filter_keeps_original_fallback(self):
        raw=self.raw.copy();raw[:,0]=4;r=self.run_case(raw)
        self.assertTrue(r['fallback']);self.assertEqual(r['methods']['learned']['index'],r['methods']['true_cost_predicted_filter']['index'])
    def test_development_geometry_split(self):
        rows=d.geometries();self.assertEqual(len(rows),12);self.assertEqual(sum(q['split']=='train' for q in rows),8)
        for q in rows:self.assertGreater(np.linalg.norm(np.diff(q['starts'],axis=0)),.3)

if __name__=='__main__':unittest.main()
