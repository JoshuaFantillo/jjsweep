using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

public class MeshColliderGeneration : MonoBehaviour
{
    public void SetMeshCollider(Mesh mesh, bool isConvex) 
    {
        MeshCollider meshCollider = gameObject.AddComponent<MeshCollider>();    
        meshCollider.sharedMesh = mesh;
        meshCollider.convex = isConvex;
    }
}
