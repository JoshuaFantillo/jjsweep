using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RandomForceApply : MonoBehaviour
{
    [HideInInspector]
    public float interval;
    [HideInInspector]
    public float directionMagnitude;
    [HideInInspector] 
    public float forceMagnitude;
    [HideInInspector]
    public bool canStart = false;

    private double lastForceApplyStamp;
    private Rigidbody rigidbody;

    private void Start()
    {
        lastForceApplyStamp = 0f;
        rigidbody = GetComponent<Rigidbody>();  
    }
    private void FixedUpdate()
    {
        if(canStart == false)
        {
            return;
        }

        lastForceApplyStamp += Time.deltaTime;
        if (lastForceApplyStamp > interval)
        {
            ApplyRandomForce();
            lastForceApplyStamp = 0f;
        }
    }

    private void ApplyRandomForce() 
    {
        float randomX = Random.Range(-directionMagnitude, directionMagnitude);
        float randomY = Random.Range(-directionMagnitude, directionMagnitude);
        float randomZ = Random.Range(-directionMagnitude, directionMagnitude);

        Vector3 direction = new Vector3(randomX, randomY, randomZ); 
        Vector3 directionNorm = direction.normalized;
        Vector3 force = directionNorm * forceMagnitude;

        Debug.Log($"Applying force {force}");
        rigidbody.AddForce(force);
    }
}
