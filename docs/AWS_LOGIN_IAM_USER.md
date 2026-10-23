# Alternative login: an IAM user with an access key

Use this **instead of Part 3 and Part 5** of [FIRST_AWS_RUN.md](FIRST_AWS_RUN.md) if IAM Identity Center
feels too long. It works, but it is **less safe**: an access key is a long-lived secret stored on your laptop.
If it leaks, someone can use your account until you delete it. Read the safety rules at the end first.

You still need Part 1 (account + root MFA) and Part 2 (budget alert) from the main guide, and Part 4 (install tools).

## A. Create the user (10 minutes)

1. Sign in to the console as the **root user** one last time.
2. Search **IAM** > left menu **Users** > **Create user**.
3. User name: `bp-admin`. Tick **Provide user access to the AWS Management Console**.
   Choose **I want to create an IAM user**, then **Custom password**, type a strong password (use a password
   manager), and keep **Users must create a new password at next sign-in** ticked. Next.
4. **Attach policies directly** > search and tick **AdministratorAccess** > Next > **Create user**.
   (Fine for a personal learning account. The Terraform in this project creates IAM roles, which needs
   these rights. Do not do this in a work account without permission.)
5. On the confirmation page, copy the **console sign-in URL**
   (`https://123456789012.signin.aws.amazon.com/console`). Save it as a bookmark.

## B. Protect the user with MFA (5 minutes)

1. Open the sign-in URL in a **private window**, sign in as `bp-admin`, and set the new password when asked.
2. Click your name (top right) > **Security credentials** > **Multi-factor authentication (MFA)** >
   **Assign MFA device** > authenticator app > scan the QR code > enter two consecutive codes.
3. Sign out and back in to prove the MFA works. From now on use `bp-admin` and not root.

## C. Create an access key for the terminal (5 minutes)

1. Signed in as `bp-admin`: **Security credentials** > **Access keys** > **Create access key**.
2. Use case: **Command Line Interface (CLI)**. Tick the confirmation box. Next.
3. Description: `laptop learning`. Create.
4. The next screen shows the **Access key ID** and the **Secret access key**. The secret is shown **only
   once**. Keep this page open; do **not** screenshot it, email it, or paste it anywhere except Terminal.

## D. Put the key on your laptop safely (5 minutes)

In Terminal, type (not paste into a file or chat):

```bash
aws configure --profile bp
```

It asks four questions:

* **AWS Access Key ID:** paste the key ID
* **AWS Secret Access Key:** paste the secret
* **Default region name:** `eu-west-2`
* **Default output format:** `json`

Then lock the file and test:

```bash
chmod 600 ~/.aws/credentials
export AWS_PROFILE=bp
export AWS_REGION=eu-west-2
aws sts get-caller-identity
```

Check: it prints your 12-digit account number and a user ARN ending in `user/bp-admin`.

Now go back to **Part 6** of FIRST_AWS_RUN.md. There is no "log in again" step with this route, because the key
does not expire. You must still run the two `export` lines in every new Terminal window.

## Safety rules (please follow all of them)

1. **Never** put the key in a file inside a git folder, a screenshot, an email, a chat message, or a notebook.
   (This project's `.gitignore` blocks `.env`, but only if you do not name the file differently.)
2. **When you finish a session or a project, delete or deactivate the key:**
   IAM > Users > `bp-admin` > **Security credentials** > **Access keys** > **Actions** > **Deactivate**, then **Delete**.
   Create a new one next time. This takes two minutes and removes most of the risk.
3. If you think the key leaked (for example you pasted it somewhere by mistake): **delete it immediately**, then
   open **Billing** and **EC2** and check for resources you did not create, and tell AWS Support if you see any.
4. Do not share the key with anyone, including an AI assistant. If a tool needs access, give it a separate,
   narrow, temporary user instead.
5. Keep the budget alert from Part 2 on. Strange charges are often the first sign of a leaked key.
6. Consider moving to IAM Identity Center later (Part 3 of the main guide). It uses temporary logins, so there
   is no long-lived secret to lose.

## Troubleshooting

| Message | Meaning | Fix |
| --- | --- | --- |
| `InvalidClientTokenId` or `SignatureDoesNotMatch` | the key or secret was mistyped | run `aws configure --profile bp` again, check for spaces |
| `AccessDenied` for IAM or SageMaker actions | the user is missing AdministratorAccess | IAM > Users > `bp-admin` > Permissions |
| `Unable to locate credentials` | the profile is not selected | `export AWS_PROFILE=bp` |
| Console says "MFA required" | you skipped step B | add an MFA device |
