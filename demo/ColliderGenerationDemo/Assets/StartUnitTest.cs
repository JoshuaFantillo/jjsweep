using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class StartUnitTest : MonoBehaviour
{
    [SerializeField]
    private Transform dropPosition;

    [SerializeField]
    [Range(0f, 5f)]
    public float Interval;

    [SerializeField]
    [Range(0f, 50f)]
    public float DirectionMagnitude;

    [SerializeField]
    [Range(-1000f, 1000f)]
    public float ForceMagnitude;

    public void StartTest(GameObject testObject)
    {
        GameObject clone = Instantiate(testObject, dropPosition.position, Quaternion.identity);
        RandomForceApply forceApply = clone.AddComponent<RandomForceApply>();
        forceApply.interval = Interval;
        forceApply.directionMagnitude = DirectionMagnitude;
        forceApply.forceMagnitude = ForceMagnitude;
        forceApply.canStart = true; 
    }
}
