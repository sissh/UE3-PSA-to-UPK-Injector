#!/usr/bin/env python3
"""Lightweight regression tests for UE3 PSA normalization (stdlib only)."""
import math
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from ue3_psa_upk_injector import PSA, Unsupported, normalize_psa, q_angle


def chunk(name, data, size, count):
    return struct.pack('<20siii', name.encode('ascii'), 2003321, size, count) + data


def sample_file(path, bones, animations, angle=0, bad_translation=False):
    names=b''.join(name.encode().ljust(64,b'\0') + b'\0'*56 for name in bones)
    meta=bytearray();keys=bytearray();first=0
    for anim,frames,rate in animations:
        rawname=anim.encode().ljust(64,b'\0')
        meta.extend(struct.pack('<64s64siiiiiffiii',rawname,b'\0'*64,len(bones),0,0,0,0,float(frames),float(rate),0,first,frames))
        for f in range(frames):
            for b in bones:
                # Artificially modify rider only, in "ride", and only after frame zero.
                degrees = angle if b=='rider' and anim=='ride' and f>=1 else 0
                q=(0.,math.sin(math.radians(degrees)/2),0.,math.cos(math.radians(degrees)/2))
                p=(2.0 if bad_translation and b=='rider' and anim=='ride' else 0.,0.,0.)
                keys.extend(struct.pack('<3f4ff',*p,*q,1./30))
        first+=frames
    path.write_bytes(chunk('ANIMHEAD',b'',0,0)+chunk('BONENAMES',names,120,len(bones))+
                     chunk('ANIMINFO',meta,168,len(animations))+
                     chunk('ANIMKEYS',keys,32,len(keys)//32))
    return PSA(path)

class NormalizeTests(unittest.TestCase):
    def test_reorder_subsets_extra_preserve_metadata(self):
        with TemporaryDirectory() as td:
            root=Path(td)
            a=sample_file(root/'orig.psa',['root','rider','animal'],[('idle',2,18),('ride',3,30)])
            b=sample_file(root/'mod.psa',['rider','root','extra'],[('ride',3,24)],angle=80)
            c,changed,moved,report=normalize_psa(a,b,0.05)
            self.assertEqual(report['missing_bones_restored'],['animal'])
            self.assertEqual(report['extra_modified_bones_ignored'],['extra'])
            self.assertTrue(report['bone_order_different'])
            self.assertEqual(report['missing_animations_preserved'],['idle'])
            self.assertEqual(report['rate_mismatches']['ride']['reference'],30)
            self.assertEqual(set(changed),{'ride'})
            self.assertEqual(set(changed['ride']),{'rider'})
            self.assertEqual(changed['ride']['rider']['changed_frames'],2)
            self.assertFalse(moved)
            self.assertEqual(a.quaternion('idle',1,'animal'),c.quaternion('idle',1,'animal'))
            self.assertEqual(a.quaternion('ride',2,'animal'),c.quaternion('ride',2,'animal'))
            self.assertAlmostEqual(q_angle(c.quaternion('ride',2,'rider'),b.quaternion('ride',2,'rider')),0,places=4)
            self.assertEqual(c.animations['ride'].rate,30)
            self.assertEqual(c.bones,a.bones)

    def test_frame_count_mismatch_rejected(self):
        with TemporaryDirectory() as td:
            root=Path(td)
            a=sample_file(root/'orig.psa',['root','rider'],[('ride',3,30)])
            b=sample_file(root/'mod.psa',['rider'],[('ride',4,24)],angle=40)
            with self.assertRaisesRegex(Unsupported,'Frame count differs'):
                normalize_psa(a,b)

    def test_translation_mismatch_reported(self):
        with TemporaryDirectory() as td:
            root=Path(td)
            a=sample_file(root/'orig.psa',['root','rider'],[('ride',3,30)])
            b=sample_file(root/'mod.psa',['rider','root'],[('ride',3,30)],angle=40,bad_translation=True)
            c,changes,moves,report=normalize_psa(a,b)
            self.assertEqual(moves['ride']['rider'],3)
            self.assertEqual(c.translation('ride',1,'rider'),a.translation('ride',1,'rider'))

if __name__=='__main__':
    unittest.main(verbosity=2)
