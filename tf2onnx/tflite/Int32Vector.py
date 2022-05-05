# SPDX-License-Identifier: Apache-2.0

# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Int32Vector(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Int32Vector()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsInt32Vector(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def Int32VectorBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # Int32Vector
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Int32Vector
    def Values(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # Int32Vector
    def ValuesAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # Int32Vector
    def ValuesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Int32Vector
    def ValuesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

def Start(builder): builder.StartObject(1)
def Int32VectorStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddValues(builder, values): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(values), 0)
def Int32VectorAddValues(builder, values):
    """This method is deprecated. Please switch to AddValues."""
    return AddValues(builder, values)
def StartValuesVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def Int32VectorStartValuesVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartValuesVector(builder, numElems)
def End(builder): return builder.EndObject()
def Int32VectorEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)