using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ShootingSystem : MonoBehaviour
{
    public GameObject projectilePrefab;
    public Transform shootPoint;
    public float projectileSpeed = 10f;

    // Update is called once per frame
    void Update()
    {

        // check for key press then shoot
        if(Input.GetKeyDown(KeyCode.KeypadEnter))
        {
            Shoot();
        }

        
    }

    private void Shoot()
    {
        // Instantiate projectile at shoot point
        GameObject projectile = Instantiate(projectilePrefab, shootPoint.position, shootPoint.rotation);

        // get rigidbody
        Rigidbody projectileRB = projectile.GetComponent<Rigidbody>();

        // apply force
        projectileRB.AddForce(shootPoint.forward * projectileSpeed, ForceMode.VelocityChange);
    }
}
