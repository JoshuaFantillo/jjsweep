using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GeneratedMeshInstance : MonoBehaviour
{
    [SerializeField]
    public MeshColliderGeneration MeshColliderGeneration;

    public void OnReceivedMesh(Mesh mesh, bool isConvex = false, bool isMesh = false, bool isCollider = false)
    {
        if(isCollider)
        {
            MeshColliderGeneration.SetMeshCollider(mesh, isConvex);
        }
        if(isMesh) 
        {
            MeshFilter meshFilter = gameObject.AddComponent<MeshFilter>();
            meshFilter.mesh = mesh;
        }
    }
}
