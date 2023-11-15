using System.Collections;
using System.Collections.Generic;
using UnityEngine;

<<<<<<< HEAD
public class SphereMovement : MonoBehaviour
=======
public class NewBehaviourScript : MonoBehaviour
>>>>>>> c6f759b04eb3c68a230c2111bddcd2ef63bc85a7
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
<<<<<<< HEAD
        
=======
        if (Input.GetKeyDown(KeyCode.UpArrow))
        {
            transform.Translate((Vector3.forward * Time.deltaTime) * 40);
        }
        if (Input.GetKeyDown(KeyCode.DownArrow))
        {
            transform.Translate((Vector3.back * Time.deltaTime) * 40);
        }
        if (Input.GetKeyDown(KeyCode.LeftArrow))
        {
            transform.Translate((Vector3.left * Time.deltaTime) * 40);
        }
        if (Input.GetKeyDown(KeyCode.RightArrow))
        {
            transform.Translate((Vector3.right * Time.deltaTime) * 40);
        }


        // Move the object upward in world space 1 unit/second.
        transform.Translate(Vector3.up * Time.deltaTime, Space.World);
>>>>>>> c6f759b04eb3c68a230c2111bddcd2ef63bc85a7
    }
}
