using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CubeController : MonoBehaviour
{

    public GameObject premadeCube;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        // Check for a key press to instantiate the cube
        if (Input.GetKeyDown(KeyCode.Space))
        {
            // Instantiate the cube prefab
            Instantiate(premadeCube, transform.position, Quaternion.identity);
        }
    }
}
