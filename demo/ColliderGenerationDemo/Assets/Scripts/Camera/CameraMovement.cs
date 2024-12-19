using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraMovement : MonoBehaviour
{
    [SerializeField]
    [Range(0f, 10f)]
    private float speed;

    void Update()
    {
        Vector3 forwardVector = Vector3.zero;
        Vector3 rightVector = Vector3.zero;
        if (Input.GetKey(KeyCode.W))
        {
            forwardVector = transform.forward;
        }

        if (Input.GetKey(KeyCode.A))
        {
            rightVector = -transform.right;
        }

        if (Input.GetKey(KeyCode.S))
        {
            forwardVector = -transform.forward;
        }
        if (Input.GetKey(KeyCode.D))
        {
            rightVector = transform.right; 
        }

        transform.position = transform.position + ((forwardVector + rightVector) * Time.deltaTime * speed);
    }

}
