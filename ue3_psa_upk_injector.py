#!/usr/bin/env python3
"""PSA-diff -> UE3 UPK AnimSequence injector with automatic PSA normalization.

Designed for uncompressed UE3 packages whose AnimSequence rotations use:
    RotationCompressionFormat = ACF_Float96NoW
    KeyEncodingFormat = AKF_VariableKeyLerp
    float XYZ rotations plus a uint8 frame number per variable key.

The user chooses the TARGET AnimSequence. No game-specific offsets, hashes,
export indices or bone IDs are hardcoded. Modified Blender PSA bone lists can
be incomplete or reordered: modified rotations are automatically rebased by
(animation name, frame index, bone name) onto the original PSA. Missing bones
and animations retain the original keys. Only rotation keys are patched in
the UPK; all other exports, positions, timings, metadata and size are preserved.

This tool supports constant rotation tracks containing a single compressed key
automatically. A varying rotation cannot fit in a single key without
rebuilding the entire compressed stream. Such tracks are SKIPPED by default
(partial injection with an explicit warning and JSON report). Set
--single-key-policy error to reject an incomplete injection instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

SIGNATURE = b'\xc1\x83\x2a\x9e'

class Unsupported(ValueError):
    """The input is unsupported or ambiguous: refuse to modify it."""

def require(cond, message):
    if not cond:
        raise Unsupported(message)

def u32(b, off):
    return struct.unpack_from('<I', b, off)[0]

def i32(b, off):
    return struct.unpack_from('<i', b, off)[0]

def sha256(b):
    return hashlib.sha256(b).hexdigest()

def q_unit(q):
    n = math.sqrt(sum(v*v for v in q))
    require(math.isfinite(n) and n > 1.e-8, 'Quaternion invalid or zero')
    return tuple(v/n for v in q)

def q_dot(a, b):
    return sum(x*y for x,y in zip(a,b))

def q_angle(a, b):
    return math.degrees(2*math.acos(min(1., max(-1.,abs(q_dot(q_unit(a),q_unit(b)))))))

def q_mul(a,b):
    x,y,z,w=a; X,Y,Z,W=b
    return (w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X, w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z)

def q_inv(a):
    x,y,z,w=q_unit(a)
    return (-x,-y,-z,w)

def q_slerp(a,b,t):
    a=q_unit(a);b=q_unit(b)
    t=min(1.,max(0.,t));d=q_dot(a,b)
    if d < 0:
        b=tuple(-v for v in b);d=-d
    d=min(1.,max(-1.,d))
    if d>.9995:
        return q_unit(tuple(x*(1-t)+y*t for x,y in zip(a,b)))
    theta=math.acos(d);den=math.sin(theta)
    return q_unit(tuple((math.sin((1-t)*theta)*x+math.sin(t*theta)*y)/den for x,y in zip(a,b)))

def psa_to_ue3(q):
    """PSA -> UE3 cooking convention verified on sample Paladins/Jenos data.

    NOTE: some UE3 game variants may use another quaternion convention. This
    must be revalidated on an original sequence before deploying to new titles.
    """
    x,y,z,w=q_unit(q)
    sign=1. if w<0 else -1.
    return q_unit((sign*x,-sign*y,sign*z,abs(w)))

def ue3_xyz_decode(b,off):
    xyz=struct.unpack_from('<fff',b,off)
    sq=sum(x*x for x in xyz)
    require(all(math.isfinite(x) for x in xyz) and sq<=1.0001, f'Invalid Float96NoW quaternion at {off:#x}: {xyz}')
    return q_unit((*xyz,math.sqrt(max(0.,1.-sq))))

def ue3_xyz_encode(b,off,q):
    q=q_unit(q)
    if q[3] < 0:
        q=tuple(-v for v in q)
    require(sum(x*x for x in q[:3])<1.000001, 'Quaternion exceeds Float96NoW bounds')
    struct.pack_into('<fff',b,off,*q[:3])

@dataclass(frozen=True)
class PSAAnimation:
    name: str
    first: int
    frames: int
    rate: float

class PSA:
    def __init__(self, path, raw=None):
        self.path=Path(path)
        self.raw=self.path.read_bytes() if raw is None else bytes(raw)
        self.chunks={}
        p=0
        while p<len(self.raw):
            require(p+32<=len(self.raw), 'Truncated PSA chunk header')
            head,version,stride,num=struct.unpack_from('<20siii',self.raw,p)
            name=head.split(b'\0',1)[0].decode('ascii','replace')
            require(name not in self.chunks and stride>=0 and num>=0, f'Invalid PSA chunk {name}')
            end=p+32+stride*num
            require(end<=len(self.raw), f'Truncated PSA chunk {name}')
            self.chunks[name]=(p+32,stride,num)
            p=end
        for name in ('BONENAMES','ANIMINFO','ANIMKEYS'):
            require(name in self.chunks, f'Missing PSA {name}')
        bp,bs,bn=self.chunks['BONENAMES'];ap,ass,an=self.chunks['ANIMINFO'];kp,ks,kn=self.chunks['ANIMKEYS']
        require(bs==120 and ass==168 and ks==32, 'Unsupported PSA record size')
        self.bones=[self.raw[bp+i*bs:bp+i*bs+64].split(b'\0')[0].decode('utf-8','replace') for i in range(bn)]
        require(len(set(self.bones))==bn, 'Duplicate bone names in PSA')
        self.bone_index={k:i for i,k in enumerate(self.bones)}
        self.animations={}
        for i in range(an):
            vals=struct.unpack_from('<64s64siiiiiffiii',self.raw,ap+i*ass)
            name=vals[0].split(b'\0')[0].decode('utf-8','replace')
            total_bones,first,frames,rate=vals[2],vals[-2],vals[-1],vals[-4]
            require(total_bones==bn and first>=0 and frames>=1 and (first+frames)*bn<=kn, f'Invalid PSA AnimInfo: {name}')
            require(name not in self.animations, f'Duplicate PSA animation: {name}')
            self.animations[name]=PSAAnimation(name,first,frames,rate)
        self.key_start=kp
        self.bone_count=bn

    def quaternion(self, anim, frame, bone):
        anim=self.animations[anim] if isinstance(anim,str) else anim
        require(0<=frame<anim.frames, 'PSA frame outside animation')
        i=self.bone_index[bone]
        off=self.key_start+(anim.first+frame)*self.bone_count*32+i*32+12
        return q_unit(struct.unpack_from('<4f',self.raw,off))

    def translation(self, anim, frame, bone):
        anim=self.animations[anim] if isinstance(anim,str) else anim
        off=self.key_start+(anim.first+frame)*self.bone_count*32+self.bone_index[bone]*32
        return struct.unpack_from('<3f',self.raw,off)

    def sample(self, anim, bone, position):
        a=self.animations[anim]
        t=min(1.,max(0.,position))*(a.frames-1)
        f=min(a.frames-1,int(math.floor(t)))
        g=min(f+1,a.frames-1)
        return q_slerp(self.quaternion(a,f,bone),self.quaternion(a,g,bone),t-f)


def normalize_psa(original, modified, angle_epsilon=0.05, position_epsilon=0.001):
    """Rebase changed rotations on the full reference PSA, in memory.

    Matches animations and bones by NAME, frames by INDEX. Restores absent
    bones/animations from original (e.g. an animal missing from the rider PSK).
    Preserves every other reference byte, including bone hierarchy, FPS,
    key order, translations, and all unmodified rotations.

    No assumption that Blender used the original bone or animation order.
    Bone local coordinates MUST still be compatible; no skeletal retargeting.
    """
    require(math.isfinite(angle_epsilon) and angle_epsilon>=0, 'Invalid angular tolerance')
    require(math.isfinite(position_epsilon) and position_epsilon>=0, 'Invalid positional tolerance')
    old_names=set(original.bones); new_names=set(modified.bones)
    shared=[b for b in original.bones if b in new_names]
    require(shared, 'Original and modified PSA have no bones in common')
    anims=[n for n in original.animations if n in modified.animations]
    require(anims, 'Original and modified PSA have no animations in common')
    for name in anims:
        aa=original.animations[name];bb=modified.animations[name]
        require(aa.frames==bb.frames,
                f'Frame count differs in {name}: {aa.frames} (original) vs '
                f'{bb.frames} (Blender); cannot safely normalize by frame index')
    rebased=bytearray(original.raw)
    changed={}; moved={}; modified_key_count=0
    changed_animation_count=0
    for name in anims:
        aa=original.animations[name];bb=modified.animations[name]
        for bone in shared:
            nrot=0; npos=0; maximum=0.
            old_idx=original.bone_index[bone]
            new_idx=modified.bone_index[bone]
            for fr in range(aa.frames):
                p0=original.key_start+((aa.first+fr)*original.bone_count+old_idx)*32
                p1=modified.key_start+((bb.first+fr)*modified.bone_count+new_idx)*32
                xyz0=struct.unpack_from('<3f',original.raw,p0)
                xyz1=struct.unpack_from('<3f',modified.raw,p1)
                require(all(math.isfinite(v) for v in (*xyz0,*xyz1)),
                        f'Invalid PSA translation in {name}/{bone}, frame {fr}')
                if max(abs(x-y) for x,y in zip(xyz0,xyz1))>position_epsilon:
                    npos+=1
                q0=struct.unpack_from('<4f',original.raw,p0+12)
                q1=struct.unpack_from('<4f',modified.raw,p1+12)
                diff=q_angle(q0,q1)
                if diff>angle_epsilon:
                    rebased[p0+12:p0+28]=modified.raw[p1+12:p1+28]
                    nrot+=1
                    maximum=max(maximum,diff)
            if nrot:
                changed.setdefault(name,{})[bone]={
                    'changed_frames':nrot,'max_angle_deg':round(maximum,4)}
                modified_key_count+=nrot
            if npos:
                moved.setdefault(name,{})[bone]=npos
    result=PSA(original.path,raw=rebased)
    require(result.bones==original.bones and result.animations==original.animations
            and result.chunks==original.chunks,
            'Normalization altered reference PSA structure')
    info={
        'applied':original.bones!=modified.bones or
                  list(original.animations)!=list(modified.animations) or
                  any(original.animations[n].rate!=modified.animations[n].rate for n in anims),
        'original_bone_count':len(original.bones),
        'modified_bone_count':len(modified.bones),
        'matched_bones':len(shared),
        'missing_bones_restored':[b for b in original.bones if b not in new_names],
        'extra_modified_bones_ignored':[b for b in modified.bones if b not in old_names],
        'bone_order_different':[b for b in modified.bones if b in old_names]!=shared,
        'original_animation_count':len(original.animations),
        'modified_animation_count':len(modified.animations),
        'missing_animations_preserved':[n for n in original.animations if n not in modified.animations],
        'extra_modified_animations_ignored':[n for n in modified.animations if n not in original.animations],
        'animation_order_different':[n for n in modified.animations if n in original.animations]!=anims,
        'rate_mismatches':{n:{'reference':original.animations[n].rate,
                              'modified':modified.animations[n].rate}
                           for n in anims if abs(original.animations[n].rate-modified.animations[n].rate)>0.00001},
        'reference_metadata_and_timing_preserved':True,
        'reference_translations_preserved':True,
        'rotations_transferred':modified_key_count,
        'normalized_sha256':sha256(result.raw),
    }
    return result,changed,moved,info


def compare_psa(original,modified,angle_epsilon=0.05):
    """Backward-compatible comparison, now tolerates subsets/reordered bones."""
    _,changed,moved,_=normalize_psa(original,modified,angle_epsilon)
    return changed,moved

@dataclass(frozen=True)
class Export:
    index: int
    classindex: int
    outer: int
    name: str
    name_number: int
    offset: int
    size: int

class UPK:
    def __init__(self,path):
        self.path=Path(path)
        self.raw=self.path.read_bytes()
        b=self.raw
        require(b[:4]==SIGNATURE, 'Not an uncompressed UE3 UPK signature')
        p=12
        require(len(b)>128, 'Truncated UPK')
        ns=i32(b,p);p+=4
        require(-4096<ns<4096, 'Unreasonable folder name length')
        p+=(ns if ns>=0 else -2*ns)
        self.flags=u32(b,p);p+=4
        nc,no,ec,eo,ic,io,deps=struct.unpack_from('<7I',b,p)
        require(0<nc<5_000_000 and 0<ec<1_000_000 and ic<1_000_000, 'Unreasonable UPK table counts')
        require(0<no<len(b) and 0<io<len(b) and 0<eo<len(b) and 0<deps<=len(b), 'UPK table offsets outside file')
        self.names=[];p=no
        for _ in range(nc):
            require(p+4<=len(b),'Truncated UPK names')
            ln=i32(b,p);p+=4
            require(ln!=0 and abs(ln)<10000,'Unsupported UE3 name string length')
            nb=ln if ln>0 else -2*ln
            require(p+nb+8<=len(b),'Truncated UPK name')
            rawname=b[p:p+nb];p+=nb
            self.names.append((rawname.split(b'\0',1)[0].decode('utf-8','replace') if ln>0
                               else rawname.decode('utf-16le','replace').rstrip('\x00')))
            p+=8
        require(p<=io, 'Name and import tables overlap')
        self.imports=[]
        for idx in range(ic):
            pos=io+idx*28
            require(pos+28<=eo,'Unsupported import entry size')
            cp,cpnum,cls,clsnum,outer,nam,num=struct.unpack_from('<7i',b,pos)
            require(all(0<=x<nc for x in (cp,cls,nam)), 'Invalid import name index')
            self.imports.append({'name':self.names[nam], 'class':self.names[cls], 'outer':outer})
        require(io+ic*28==eo, 'Unsupported import table variant')
        self.exports=[];p=eo
        for idx in range(1,ec+1):
            require(p+68<=deps, 'Truncated export table')
            cl,sup,outer,nam,nno,arch=struct.unpack_from('<6i',b,p)
            size,off,flags,net=struct.unpack_from('<4i',b,p+32)
            require(-ic<=cl<=ec and -ic<=outer<=ec and 0<=nam<nc and net>=0 and net<100000,
                    f'Unsupported export entry at {idx}')
            require(0<=size<=len(b) and 0<=off<=len(b) and off+size<=len(b),
                    f'Invalid export offset/length at {idx}')
            self.exports.append(Export(idx,cl,outer,self.names[nam],nno,off,size))
            p+=68+4*net
        require(p==deps, f'Export table cannot be parsed safely: {p:#x}!={deps:#x}')
        self.sequences=[]
        for e in self.exports:
            if self.class_name(e)=='AnimSequence':
                try:
                    props,_=self.properties(e)
                    seqname=self.fname_prop(props,'SequenceName')
                    parent=self.exports[e.outer-1].name if e.outer>0 else ''
                    self.sequences.append({'id':e.index,'sequence':seqname,'animset':parent,'export':e.name,
                                           'frames':self.int_prop(props,'NumFrames'), 'offset':e.offset,'size':e.size})
                except (KeyError,ValueError,IndexError,struct.error,Unsupported):
                    # Some variants may be opaque; never select without a readable name.
                    continue

    def class_name(self,e):
        if e.classindex<0:
            return self.imports[-e.classindex-1]['name']
        if e.classindex>0:
            return self.exports[e.classindex-1].name
        return ''

    def properties(self,e):
        b=self.raw;p=e.offset+4;end=e.offset+e.size;d={};n=0
        require(e.size>=12, 'Export too small for UE3 properties')
        while p<end and n<10000:
            require(p+8<=end, 'Truncated property name')
            ni,number=struct.unpack_from('<II',b,p);p+=8
            require(ni<len(self.names), 'Property name index outside names table')
            name=self.names[ni]
            if name=='None':
                return d,p
            require(p+16<=end, 'Truncated property metadata')
            ti,tno,sz,idx=struct.unpack_from('<4I',b,p);p+=16
            require(ti<len(self.names),'Property type index outside names table')
            typ=self.names[ti]
            if typ in ('ByteProperty','StructProperty'):p+=8
            if typ=='BoolProperty':p+=1
            require(p+sz<=end, f'Property {name} escapes serialized export')
            d[name]=(p,sz,typ)
            p+=sz;n+=1
        raise Unsupported('Cannot locate end of property list')

    def fname_prop(self,props,key):
        off,sz,typ=props[key]
        require(sz==8 and typ in ('NameProperty','ByteProperty'), f'{key} is not an FName/Byte enum')
        idx,num=struct.unpack_from('<II',self.raw,off)
        require(idx<len(self.names),f'Invalid FName index for {key}')
        return self.names[idx]

    def int_prop(self,props,key):
        off,sz,typ=props[key]
        require(sz==4, f'{key} is not 4 bytes')
        return i32(self.raw,off)

    def select(self,sequence=None,animset=None,export_id=None):
        candidates=self.sequences
        if export_id is not None:
            candidates=[x for x in candidates if x['id']==export_id]
        if sequence is not None:
            candidates=[x for x in candidates if x['sequence'].lower()==sequence.lower()]
        if animset is not None:
            candidates=[x for x in candidates if x['animset'].lower()==animset.lower()]
        if len(candidates)!=1:
            d='\n'.join(f"  {x['id']} : {x['animset']} / {x['sequence']} ({x['frames']} frames)" for x in candidates[:30])
            raise Unsupported(f'{len(candidates)} matching AnimSequences; specify --sequence, --animset, or --export-id.\n{d}')
        return candidates[0]

    def track_names(self,target):
        e=self.exports[target['id']-1]
        require(e.outer>0, 'AnimSequence has no parent AnimSet')
        animset=self.exports[e.outer-1]
        require(self.class_name(animset)=='AnimSet', 'AnimSequence outer is not an AnimSet')
        props,tail=self.properties(animset)
        off,sz,typ=props['TrackBoneNames']
        n=u32(self.raw,off)
        require(n>0 and sz==4+n*8, 'Unexpected AnimSet TrackBoneNames encoding')
        bones=[]
        for i in range(n):
            ix,number=struct.unpack_from('<II',self.raw,off+4+i*8)
            require(ix<len(self.names) and number==0, 'Unsupported named/suffixed track')
            bones.append(self.names[ix])
        require(len(set(bones))==len(bones),'Duplicate track bone names')
        return bones

    def compressed(self,target,track_count):
        e=self.exports[target['id']-1]
        props,tail=self.properties(e)
        require(self.fname_prop(props,'RotationCompressionFormat')=='ACF_Float96NoW',
                'Codec unsupported: only ACF_Float96NoW')
        require(self.fname_prop(props,'KeyEncodingFormat')=='AKF_VariableKeyLerp',
                'Key encoding unsupported: only AKF_VariableKeyLerp')
        nf=self.int_prop(props,'NumFrames')
        require(1<nf<=255, 'Frame indices >=256 need a different key-time reader')
        po,sz,typ=props['CompressedTrackOffsets']
        require(sz==4+16*track_count and u32(self.raw,po)==4*track_count,
                'Unexpected CompressedTrackOffsets table')
        require(tail+8<=e.offset+e.size and self.raw[tail:tail+4]==b'\0\0\0\0',
                'Unsupported CompressedByteStream header')
        blen=u32(self.raw,tail+4)
        body=tail+8
        require(body+blen==e.offset+e.size, 'Compressed stream not export-aligned')
        return (nf,po,body,blen)

    def rotation_track(self,record,ti,comp):
        nf,po,body,blen=comp
        position_keys,position_count,roff,rn=struct.unpack_from('<4i',self.raw,po+4+16*ti)
        require(1<=rn<=nf, f'Rotation track {ti}: {rn} keys for {nf} frames; '
                'zero-key and oversampled tracks unsupported')
        # A constant (1-key) UE3 track contains 12 bytes of Float96NoW only;
        # unlike a multi-key track it has NO accompanying frame indices.
        require(0<=roff and roff+12*rn+(rn if rn>1 else 0)<=blen,
                f'Rotation track {ti} extends outside compressed stream')
        if rn==1:
            ue3_xyz_decode(self.raw,body+roff)
            return [(0,body+roff)]
        raw_times=self.raw[body+roff+rn*12:body+roff+rn*12+rn]
        times=list(raw_times)
        require(times==sorted(set(times)) and times[0]==0 and times[-1]==nf-1,
                f'Non-monotone/unsupported key frame indices on track {ti}')
        for i in range(rn):
            ue3_xyz_decode(self.raw,body+roff+i*12)
        return [(frame,body+roff+i*12) for i,frame in enumerate(times)]


def proposed_rotation(oldq, psa_original, psa_modified, anim, bone, t, mode, strength):
    """Compute the final local target rotation at normalized animation time t."""
    org=psa_to_ue3(psa_original.sample(anim,bone,t))
    new=psa_to_ue3(psa_modified.sample(anim,bone,t))
    target=(q_unit(q_mul(oldq,q_mul(q_inv(org),new)))
            if mode=='delta' else new)
    return q_slerp(oldq,target,strength) if strength!=1.0 else target


def constant_key_proposal(oldq,a,b,anim,bone,mode,strength,angular_tolerance):
    """Check that the requested target pose stays constant throughout the PSA.

    Inspect original PSA frames AND half-frames to detect a changing delta,
    which a 1-key target rotation track cannot represent.
    """
    count=a.animations[anim].frames
    samples=[i/(2*(count-1)) for i in range(2*count-1)] if count>1 else [0.0]
    proposal=proposed_rotation(oldq,a,b,anim,bone,0.0,mode,strength)
    maximum=0.0
    for t in samples[1:]:
        maximum=max(maximum,q_angle(proposal,
                            proposed_rotation(oldq,a,b,anim,bone,t,mode,strength)))
        if maximum>angular_tolerance:
            break
    return proposal,maximum


def run(opts):
    upk=UPK(opts.upk)
    print(f'UPK: {upk.path.name}; exports={len(upk.exports)}; readable sequences={len(upk.sequences)}')
    if opts.list:
        cand=[x for x in upk.sequences if not opts.filter or opts.filter.lower() in
              (x['animset']+' '+x['sequence']).lower()]
        for x in cand:
            print(f"[{x['id']:5}] {x['animset']} / {x['sequence']} ({x['frames']} frames)")
        print(f'{len(cand)} AnimSequences listed')
        return
    require(opts.psa_original and opts.psa_modified, 'Provide --psa-original and --psa-modified')
    require(opts.sequence or opts.export_id, 'Choose --sequence or --export-id (use --list to discover)')
    if opts.expected_upk_sha256:
        require(sha256(upk.raw)==opts.expected_upk_sha256.lower(), 'UPK SHA-256 mismatch')
    if opts.expected_upk_size is not None:
        require(len(upk.raw)==opts.expected_upk_size, 'UPK file size mismatch')
    a=PSA(opts.psa_original);b=PSA(opts.psa_modified)
    for data,kind in ((a.raw,'original_psa'),(b.raw,'modified_psa')):
        want_hash=getattr(opts,'expected_'+kind+'_sha256')
        want_size=getattr(opts,'expected_'+kind+'_size')
        if want_hash:
            require(sha256(data)==want_hash.lower(),f'{kind} SHA-256 mismatch')
        if want_size is not None:
            require(len(data)==want_size,f'{kind} size mismatch')
    b_raw=b
    b,changed,moved,normalization=normalize_psa(
        a,b_raw,opts.angle_epsilon,opts.position_epsilon)
    print(f"Automatic PSA normalization: {normalization['matched_bones']} matching bones, "
          f"{len(normalization['missing_bones_restored'])} restored bones, "
          f"{len(normalization['extra_modified_bones_ignored'])} extra bones ignored, "
          f"{normalization['rotations_transferred']} rotation keys transferred")
    if normalization['rate_mismatches']:
        print(f"NOTE: {len(normalization['rate_mismatches'])} animations have different FPS values; "
              "original frame rates preserved (check Blender animation timing).")
    if moved:
        detail=', '.join(f'{name}: {len(bones)} bones' for name,bones in moved.items())
        require(opts.ignore_translation_changes,
                'PSA translations changed, but injector edits ROTATIONS ONLY: '+detail+
                ' (use --ignore-translation-changes to opt in)')
    require(changed, 'No changes detected in PSA rotations')
    if opts.psa_animation:
        require(opts.psa_animation in changed, '--psa-animation does not contain modified rotations')
        animation=opts.psa_animation
    else:
        require(len(changed)==1,'Multiple modified PSA animations detected: '+', '.join(changed)+
                '. Select --psa-animation')
        animation=next(iter(changed))
    seq=upk.select(opts.sequence,opts.animset,opts.export_id)
    bone_names=upk.track_names(seq)
    modified=changed[animation]
    absent=[x for x in modified if x not in bone_names]
    require(not absent or opts.skip_missing_bones,
            f'{len(absent)} changed bones absent in target AnimSet: {absent[:20]}; '
            'use --skip-missing-bones to inject only the common tracks')
    joints=[x for x in modified if x in bone_names]
    require(joints, 'No edited PSA bones exist in target AnimSet')
    comp=upk.compressed(seq,len(bone_names))
    out=bytearray(upk.raw)
    allowed=set();statistics=[];skipped_single_key=[]
    for bone in joints:
        ti=bone_names.index(bone)
        keys=upk.rotation_track(seq,ti,comp)
        changed_keys=0;max_deg=0.0
        if len(keys)==1:
            _,pos=keys[0]
            oldq=ue3_xyz_decode(upk.raw,pos)
            target,variation=constant_key_proposal(
                oldq,a,b,animation,bone,opts.mode,opts.strength,
                opts.single_key_tolerance)
            if variation>opts.single_key_tolerance:
                skipped_single_key.append({'bone':bone,'track_index':ti,
                                           'variation_deg_at_least':round(variation,3),
                                           'reason':'single-key UPK track cannot encode changing PSA rotations'})
                continue
            if q_angle(oldq,target)>1.e-5:
                ue3_xyz_encode(out,pos,target)
                allowed.update(range(pos,pos+12))
                changed_keys=1;max_deg=q_angle(oldq,target)
            statistics.append({'bone':bone,'psa_bone_index':a.bone_index[bone],
                               'upk_track_index':ti,'target_rotation_keys':1,
                               'keys_changed':changed_keys,
                               'max_target_rotation_deg':round(max_deg,3),
                               'psa_max_change_deg':modified[bone]['max_angle_deg']})
            continue
        for frame,pos in keys:
            t=frame/(seq['frames']-1)
            org=psa_to_ue3(a.sample(animation,bone,t))
            new=psa_to_ue3(b.sample(animation,bone,t))
            diff=q_angle(org,new)
            if diff<=opts.angle_epsilon:
                continue
            oldq=ue3_xyz_decode(upk.raw,pos)
            # 'delta' retains target baseline, but assumes compatible bone local axes.
            # 'absolute' sets the sampled modified PSA pose directly in target rig.
            if opts.mode=='delta':
                delta=q_mul(q_inv(org),new)
                target=q_unit(q_mul(oldq,delta))
            else:
                target=new
            if opts.strength!=1.0:
                target=q_slerp(oldq,target,opts.strength)
            if q_angle(oldq,target)<=1.e-5:
                continue
            ue3_xyz_encode(out,pos,target)
            allowed.update(range(pos,pos+12))
            changed_keys+=1;max_deg=max(max_deg,q_angle(oldq,target))
        statistics.append({'bone':bone,'psa_bone_index':a.bone_index[bone],
                           'upk_track_index':ti,'target_rotation_keys':len(keys),
                           'keys_changed':changed_keys,
                           'max_target_rotation_deg':round(max_deg,3),
                           'psa_max_change_deg':modified[bone]['max_angle_deg']})
    if skipped_single_key and opts.single_key_policy!='skip':
        examples=', '.join(f"{x['bone']} (track {x['track_index']}, "
                           f"variation >= {x['variation_deg_at_least']:.3f} deg)"
                           for x in skipped_single_key[:12])
        require(False,
                f"{len(skipped_single_key)} modified bone(s) in '{seq['sequence']}' "
                f"have only ONE rotation key in the UPK, but "
                f"the PSA requests time-varying rotations. Examples: {examples}. "
                f"Accurate injection requires rebuilding the CompressedByteStream "
                f"and its offsets. To allow a PARTIAL injection, "
                f"use --single-key-policy skip (the default).")
    changed_positions=[i for i,(x,y) in enumerate(zip(upk.raw,out)) if x!=y]
    require(changed_positions and set(changed_positions).issubset(allowed),
            'No changed rotation keys. All candidate tracks may have been skipped '
            'or the requested rotations may already match the target.')
    ex=upk.exports[seq['id']-1]
    require(len(out)==len(upk.raw) and out[:ex.offset]==upk.raw[:ex.offset]
            and out[ex.offset+ex.size:]==upk.raw[ex.offset+ex.size:],
            'Bytes outside target AnimSequence were changed')
    report={
        'source_files':{'upk':str(upk.path),'psa_original':str(a.path),'psa_modified':str(b_raw.path)},
        'sha256':{'upk_original':sha256(upk.raw),'upk_result':sha256(out),
                  'psa_original':sha256(a.raw),'psa_modified':sha256(b_raw.raw),
                  'psa_normalized_in_memory':sha256(b.raw)},
        'size_bytes':{'upk_original':len(upk.raw),'upk_result':len(out)},
        'selected':{'id':seq['id'],'sequence':seq['sequence'],'animset':seq['animset'],
                    'frames':seq['frames'],'serial_offset':ex.offset,'serial_size':ex.size},
        'psa_animation':animation,'mode':opts.mode,'strength':opts.strength,
        'single_key_policy':opts.single_key_policy,
        'single_key_behavior':'automatic constant-key support; varying single-key tracks skipped by default and reported',
        'psa_normalization':normalization,
        'matched_bones':len(joints),'absent_bones':absent,'modified_tracks':statistics,
        'skipped_varying_single_key_tracks':skipped_single_key,
        'partial_injection':bool(skipped_single_key or (absent and opts.skip_missing_bones)),
        'changed_keys':sum(x['keys_changed'] for x in statistics),
        'changed_bytes':len(changed_positions),
        'translation_changes_ignored':moved if moved and opts.ignore_translation_changes else {},
        'checks':'PASS: same UPK length; changed bytes limited to target rotation keys; all other exports unchanged',
        'limitations':'No in-game verification; other codecs, alternate UE3 layouts and skeletal retargeting not supported'
    }
    print(f"PSA '{animation}': {len(modified)} modified bones; target export #{seq['id']} "
          f"'{seq['animset']}/{seq['sequence']}' ({seq['frames']} frames)")
    print(f"PATCH: {len(statistics)} tracks patched, {report['changed_keys']} keys, {len(changed_positions)} bytes")
    if absent: print('Skipped bones:',', '.join(absent))
    if skipped_single_key:
        print(f'WARNING: PARTIAL injection: {len(skipped_single_key)} '
              'single-key track(s) skipped: '
              ', '.join(x['bone'] for x in skipped_single_key))
    if opts.dry_run:
        print('DRY-RUN: no files written')
        return report
    # Validate optional UPK Explorer wrapper BEFORE writing any output file.
    exportpatched=None
    if opts.export_original:
        wrapper=Path(opts.export_original).read_bytes()
        blob=upk.raw[ex.offset:ex.offset+ex.size]
        position=wrapper.find(blob)
        require(position>=0 and wrapper.find(blob,position+1)<0,
                'AnimSequence export wrapper is not an exact/unique match to source export')
        exportpatched=wrapper[:position]+bytes(out[ex.offset:ex.offset+ex.size])+wrapper[position+ex.size:]
        require(len(exportpatched)==len(wrapper), 'Wrapper length changed')
    output=Path(opts.out) if opts.out else upk.path.with_name(upk.path.stem+'_patched.upk')
    require(output.resolve()!=upk.path.resolve(), 'Output path cannot overwrite source UPK')
    protected={upk.path.resolve(), a.path.resolve(), b_raw.path.resolve()}
    export_out=(Path(opts.export_out) if opts.export_out else output.with_suffix('.AnimSequence')) if exportpatched is not None else None
    report_path=Path(opts.report) if opts.report else output.with_suffix('.validation.json')
    normalized_out=Path(opts.normalized_psa_out) if opts.normalized_psa_out else None
    outputs=[output, report_path]+([export_out] if export_out is not None else [])+([normalized_out] if normalized_out is not None else [])
    require(len({x.resolve() for x in outputs})==len(outputs),'Output file paths must be distinct')
    require(all(x.resolve() not in protected for x in outputs),
            'Refusing to overwrite an input PSA or UPK file')
    if opts.export_original:
        require(all(x.resolve()!=Path(opts.export_original).resolve() for x in outputs),
                'Refusing to overwrite original AnimSequence wrapper')
    output.write_bytes(out)

    assert output.read_bytes()==out
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    if exportpatched is not None:
        export_out.write_bytes(exportpatched)
        print('ANIMSEQUENCE:',export_out)
    if normalized_out is not None:
        normalized_out.write_bytes(b.raw)
        print('NORMALIZED PSA:',normalized_out)
    print('UPK:',output)
    print('VALIDATION:',report_path)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(
        description=(
            'Inject rotation changes from a modified PSA into a chosen UE3 '
            'AnimSequence in a decompressed UPK. Missing or reordered Blender '
            'bones are normalized automatically.'
        ),
        epilog=(
            'Examples:\n'
            '  List target sequences:\n'
            '    python ue3_psa_upk_injector.py --upk original.upk --list\n'
            '  Filter the sequence list:\n'
            '    python ue3_psa_upk_injector.py --upk original.upk --list --filter astral\n'
            '  Inject modified rotations:\n'
            '    python ue3_psa_upk_injector.py --upk original.upk --psa-original original.psa --psa-modified edited.psa --sequence SequenceName --animset AnimSet --out modified.upk\n'
            '  Validate without writing files:\n'
            '    python ue3_psa_upk_injector.py --upk original.upk --psa-original original.psa --psa-modified edited.psa --sequence SequenceName --dry-run\n'
            '\n'
            'Safety: the default single-key policy is skip (partial injection is '
            'reported). Use --single-key-policy error to require complete '
            'injection. Only ACF_Float96NoW / AKF_VariableKeyLerp tracks '
            'are currently supported.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    general = p.add_argument_group('General')
    general.add_argument('-h', '--help', action='help',
                         help='Show all options, defaults, and examples; then exit')
    general.add_argument('--upk', required=True,
                         help='Source decompressed UPK (required for list/injection)')
    general.add_argument('--list', action='store_true',
                         help='List AnimSequences (only --upk is required)')
    general.add_argument('--filter',
                         help='Filter --list results by sequence or AnimSet name')

    psa = p.add_argument_group('PSA files and normalization')
    psa.add_argument('--psa-original', help='Unmodified reference PSA for rotation comparison')
    psa.add_argument('--psa-modified', help='Edited PSA exported from Blender or another tool')
    psa.add_argument('--psa-animation',
                     help='Source PSA animation name (required when several animations changed)')
    psa.add_argument('--angle-epsilon', type=float, default=0.05,
                     help='Minimum detected rotation change, degrees (default: 0.05)')
    psa.add_argument('--position-epsilon', type=float, default=0.001,
                     help='Allowed PSA translation drift (default: 0.001)')
    psa.add_argument('--ignore-translation-changes', action='store_true',
                     help='Ignore changed PSA translations; only rotations are injected')
    psa.add_argument('--normalized-psa-out',
                     help='Optionally save normalized PSA (normally in memory only)')

    target = p.add_argument_group('Target selection and rotation transfer')
    target.add_argument('--sequence', help='Name of target UPK AnimSequence')
    target.add_argument('--animset', help='Target AnimSet name, for disambiguation')
    target.add_argument('--export-id', type=int,
                        help='Select exact 1-based UPK export index instead of a sequence name')
    target.add_argument('--mode', choices=('delta', 'absolute'), default='delta',
                        help='delta applies PSA rotation changes; absolute copies modified rotations (default: delta)')
    target.add_argument('--strength', type=float, default=1.0,
                        help='Rotation strength in range 0..1 (default: 1.0)')

    safety = p.add_argument_group('Compatibility and safety')
    safety.add_argument('--skip-missing-bones', action='store_true',
                        help='Allow target bones absent from the UPK AnimSet to be skipped and reported')
    safety.add_argument('--single-key-policy', choices=('error', 'skip'), default='skip',
                        help='Varying animation on one-key tracks: skip and warn (default), or error')
    safety.add_argument('--single-key-tolerance', type=float, default=0.05,
                        help='Maximum temporal variation for one-key tracks in degrees (default: 0.05)')
    safety.add_argument('--dry-run', action='store_true',
                        help='Validate and show planned modifications without writing files')

    integrity = p.add_argument_group('Optional source integrity checks')
    integrity.add_argument('--expected-upk-sha256', help='Expected SHA-256 hash of input UPK')
    integrity.add_argument('--expected-upk-size', type=int, help='Expected input UPK size in bytes')
    integrity.add_argument('--expected-original-psa-sha256', help='Expected SHA-256 of reference PSA')
    integrity.add_argument('--expected-original-psa-size', type=int,
                           help='Expected reference PSA size in bytes')
    integrity.add_argument('--expected-modified-psa-sha256', help='Expected SHA-256 of modified PSA')
    integrity.add_argument('--expected-modified-psa-size', type=int,
                           help='Expected modified PSA size in bytes')

    outputs = p.add_argument_group('Output files')
    outputs.add_argument('--out',
                         help='Output UPK path (default: <source>_patched.upk; never overwrite source)')
    outputs.add_argument('--report',
                         help='Validation JSON path (default: output UPK with .validation.json suffix)')
    outputs.add_argument('--export-original',
                         help='Original UPK Explorer AnimSequence wrapper to patch (optional)')
    outputs.add_argument('--export-out',
                         help='Output path for patched AnimSequence wrapper')
    opts=p.parse_args(argv)
    try:
        require(math.isfinite(opts.strength) and 0<=opts.strength<=1, 'strength must be between 0 and 1')
        require(math.isfinite(opts.angle_epsilon) and opts.angle_epsilon>=0,'Invalid angle epsilon')
        require(math.isfinite(opts.position_epsilon) and opts.position_epsilon>=0,'Invalid position epsilon')
        require(math.isfinite(opts.single_key_tolerance) and opts.single_key_tolerance>=0,
                'Invalid single-key tolerance')
        run(opts)
    except (Unsupported,KeyError,struct.error,IndexError,FileNotFoundError) as ex:
        print('ERROR:',ex,file=sys.stderr)
        return 2
    return 0

if __name__=='__main__':
    sys.exit(main())
