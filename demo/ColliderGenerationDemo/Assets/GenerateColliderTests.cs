using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GenerateColliderTests : MonoBehaviour
{
    [SerializeField]
    private StartUnitTest testPrefab;

    [SerializeField]
    private int numRows = 10;

    [SerializeField] 
    private int numCols = 10;

    [SerializeField]
    private float rowOffset = 7.5f;

    [SerializeField]
    private float colOffset = 7.5f;

    private List<StartUnitTest> testLists;

    public void GenerateTests(GameObject gameObject)
    {
        float y = 0f;
        for (int i = 0; i < numRows; i++)
        {
            float x = 0f;
            for(int j = 0; j < numCols; j++)
            {
                Vector3 position = new Vector3(transform.position.x + x, transform.position.y + y, transform.position.z);
                StartUnitTest unitTest = Instantiate(testPrefab, position, Quaternion.identity);
                unitTest.StartTest(gameObject); 
                x += colOffset;
            }
            y += rowOffset;
        }
    }
}
