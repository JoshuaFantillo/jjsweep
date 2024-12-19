using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GeneratedMesh : MonoBehaviour
{
    [SerializeField]
    private GeneratedMeshInstance meshInstancePrefab;

    [SerializeField]
    public Rigidbody GeneratedMeshRigidBody;

    private GeneratedMeshInstance[] subMeshes;

    public void OnGeneratedMeshComplete(Mesh[] meshes, bool isMesh = false, bool isCollider = false, bool isConvex = true, Rigidbody rigidbody = null)
    {
        if(rigidbody == null && isConvex)
        {
            GeneratedMeshRigidBody = gameObject.AddComponent<Rigidbody>();
            GeneratedMeshRigidBody.useGravity = isConvex;
        }
        else if(rigidbody != null)
        {
            GeneratedMeshRigidBody = rigidbody;
            GeneratedMeshRigidBody.useGravity = isConvex;
        }

        for(int i = 0; i < meshes.Length; i++)
        {
            GeneratedMeshInstance meshInstance = Instantiate(meshInstancePrefab);
            meshInstance.transform.SetParent(transform, false);
            meshInstance.OnReceivedMesh(meshes[i], isConvex: isConvex, isMesh: isMesh, isCollider: isCollider);
        }
    }
}
