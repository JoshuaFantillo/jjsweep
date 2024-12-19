using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Runtime.CompilerServices;
using UnityEngine;

public class MeshFileParser : MonoBehaviour
{
    public Mesh ReadMesh(string location)
    {
        StreamReader reader = new StreamReader(location);
        Mesh mesh = CreateMeshFromReader(reader);
        reader.Close();
        return mesh;
    }

    private Mesh CreateMeshFromReader(StreamReader reader)
    {
        List<Vector3> vertices = new List<Vector3>();
        List<int> indices = new List<int>();

        while(reader.Peek() >= 0)
        {
            string data = reader.ReadLine();
            if(data != string.Empty) {
                if (data[0] == 'v') {
                    ParseVector(data, ref vertices);
                }
                else if (data[0] == 'f') {
                    ParseIndex(data, ref indices);
                }
            }
        }
        Mesh mesh = new Mesh();
        mesh.SetVertices(vertices);
        mesh.SetTriangles(indices, 0);
        return mesh;
    }

    private void ParseVector(string line, ref List<Vector3> vertices) 
    {
        string[] parsed = line.Split(' ');
        if(line.Length >= 4) {
            float x = float.Parse(parsed[1], CultureInfo.InvariantCulture.NumberFormat);
            float y = float.Parse(parsed[2], CultureInfo.InvariantCulture.NumberFormat);
            float z = float.Parse(parsed[3], CultureInfo.InvariantCulture.NumberFormat);

            vertices.Add(new Vector3(x, y, z));
        }
        
    }

    private void ParseIndex(string line, ref List<int> indices) {
        string[] parsed = line.Split(' ');
        if(parsed.Length >= 4) {
            int a = int.Parse(parsed[1], CultureInfo.InvariantCulture.NumberFormat);
            int b = int.Parse(parsed[2], CultureInfo.InvariantCulture.NumberFormat);
            int c = int.Parse(parsed[3], CultureInfo.InvariantCulture.NumberFormat);

            indices.Add(a - 1);
            indices.Add(b - 1);
            indices.Add(c - 1);
        }
    }
    
    
}
