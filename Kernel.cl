__kernel void negativeFilter(__global char* rawData, __global char* rawNegativeData)
{
    int i = get_global_id(0);
    rawNegativeData[i] = 255 - rawData[i];
}