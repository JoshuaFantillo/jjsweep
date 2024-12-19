using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraPan : MonoBehaviour
{

    [SerializeField]
    [Range(0f, 50f)]
    private float verticalSpeed;

    [SerializeField]
    [Range(0f, 50f)]
    private float horizontalSpeed;

    private float horizontalOrientation;
    private float verticalOrientation;

    private void Start()
    {
        horizontalOrientation = transform.rotation.eulerAngles.y;
        verticalOrientation = transform.rotation.eulerAngles.x;
    }

    // Update is called once per frame
    void Update()
    {
        float horizontalRotation = Input.GetAxis("Horizontal");
        float verticalRotation = Input.GetAxis("Vertical");

        horizontalOrientation += horizontalRotation * horizontalSpeed * Time.deltaTime;
        verticalOrientation -= verticalRotation * verticalSpeed * Time.deltaTime;

        verticalOrientation = Mathf.Clamp(verticalOrientation, -85f, 85f);
        transform.rotation = Quaternion.Euler(verticalOrientation, horizontalOrientation, 0.0f);
    }

}
