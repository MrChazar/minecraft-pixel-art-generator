/*
  Warnings:

  - A unique constraint covering the columns `[userEmail,action]` on the table `EmailAction` will be added. If there are existing duplicate values, this will fail.

*/
-- CreateIndex
CREATE UNIQUE INDEX "EmailAction_userEmail_action_key" ON "public"."EmailAction"("userEmail", "action");
