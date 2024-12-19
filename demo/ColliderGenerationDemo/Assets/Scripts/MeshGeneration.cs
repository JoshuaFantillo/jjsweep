using System.Collections;
using System.Collections.Generic;
using System.IO;
using Unity.VisualScripting.Antlr3.Runtime;
using UnityEngine;
using System;
using System.IO;
using System.Linq;
using Dummiesman;
using UnityEngine.WSA;
using UnityEngine.Events;

public class MeshGeneration : MonoBehaviour {


    [SerializeField]
    private MeshFileParser parser;

    [SerializeField]
    public string FileLocation = @"Assets/Resources/OriginalMesh/";

    [SerializeField]
    public string FolderLocation = @"Assets/Resources/Output/";

    [SerializeField]
    public GeneratedMesh MeshInstancePrefab;

    [SerializeField]
    public Transform MeshSpawnPoint;

    [SerializeField]
    public UnityEvent<GameObject> OnMeshComplete;

    public GameObject MeshObjectSample;
    public bool IsDynamicMeshCollisionGeneration = false;

    void Start() {
        if(IsDynamicMeshCollisionGeneration)
        {
            CreateMeshObjectDynamic(FileLocation, FolderLocation);
        }
        else
        {
            CreateMeshObjectOriginal(FileLocation);
        }

        OnMeshComplete.Invoke(MeshObjectSample);
    }

    public GameObject CreateMeshObjectDynamic(string meshLocation, string meshComponentsFolderLocation)
    {
        if (MeshObjectSample != null)
        {
            Destroy(MeshObjectSample);
            MeshObjectSample = null;    
        }

        MeshObjectSample = new GameObject();
        Rigidbody rigidBody = MeshObjectSample.AddComponent<Rigidbody>();
        GeneratedMesh meshRepresentation = ReadSingleMesh(meshLocation, null, isConvex: false, isCollider: false, isMesh: true);
        GeneratedMesh meshColliderRepresentation = ReadMultiComponentMesh(meshComponentsFolderLocation, rigidBody, isConvex: true, isCollider: true, isMesh: false);

        meshRepresentation.gameObject.transform.SetParent(MeshObjectSample.transform);
        meshColliderRepresentation.gameObject.transform.SetParent(MeshObjectSample.transform);
        MeshObjectSample.transform.position = MeshSpawnPoint.position;

        return MeshObjectSample;
    }

    public GameObject CreateMeshObjectOriginal(string meshLocation)
    {
        if (MeshObjectSample != null)
        {
            Destroy(MeshObjectSample);
            MeshObjectSample = null;
        }

        MeshObjectSample = new GameObject();
        Rigidbody rigidBody = MeshObjectSample.AddComponent<Rigidbody>();
        GeneratedMesh meshRepresentation = ReadSingleMesh(meshLocation, rigidBody, isConvex: true, isCollider: true, isMesh: true);

        meshRepresentation.gameObject.transform.SetParent(MeshObjectSample.transform);
        MeshObjectSample.transform.position = MeshSpawnPoint.position;

        return MeshObjectSample;
    }


    public GeneratedMesh ReadSingleMesh(string fileLocation, Rigidbody rigidbody, bool isConvex = false, bool isCollider = false, bool isMesh = false)
    {
        string[] filePaths = Directory.GetFiles(fileLocation)
            .Where(name => name.EndsWith(".obj"))
            .ToArray();

        OBJLoader loader = new OBJLoader();
        GameObject meshObject = loader.Load(filePaths[0]);
        Mesh mesh = meshObject.GetComponentInChildren<MeshFilter>().mesh;
        Mesh[] meshes = new Mesh[] { mesh };
        GeneratedMesh meshInstance = Instantiate(MeshInstancePrefab);
        meshInstance.OnGeneratedMeshComplete(meshes, rigidbody: rigidbody, isConvex: isConvex, isCollider: isCollider, isMesh: isMesh);
        Destroy(meshObject);
        return meshInstance;
    }

    public GeneratedMesh ReadMultiComponentMesh(string folder, Rigidbody rigidbody, bool isConvex = false, bool isCollider = false, bool isMesh = false)
    {
        string[] filePaths = Directory.GetFiles(folder)
            .Where(name => name.EndsWith(".obj"))
            .ToArray();
        List<Mesh> meshes = new List<Mesh>();   
        foreach (string filePath in filePaths)
        {
            OBJLoader loader = new OBJLoader();
            GameObject meshObject = loader.Load(filePath);
            Mesh mesh = meshObject.GetComponentInChildren<MeshFilter>().mesh;
            meshes.Add(mesh);
            Destroy(meshObject);
        }
        GeneratedMesh meshInstance = Instantiate(MeshInstancePrefab);
        meshInstance.OnGeneratedMeshComplete(meshes.ToArray(), rigidbody: rigidbody, isConvex: isConvex, isCollider: isCollider, isMesh: isMesh);
        return meshInstance;
    }

}
